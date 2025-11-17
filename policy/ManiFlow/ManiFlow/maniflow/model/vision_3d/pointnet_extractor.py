import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import timm
import os

from typing import Optional, Dict, Tuple, Union, List, Type
import maniflow.model.vision_3d.point_process as point_process
from maniflow.model.vision_3d.randlanet_encoder import LocalFeatureAggregation, SharedMLP
from maniflow.model.vision_3d.pointnet2_encoder import PointNetSetAbstraction, PointNetSetAbstractionMsg
from maniflow.model.vision_3d.fp3_encoder import Group, Encoder, PatchDropout, random_point_dropout
from termcolor import cprint

# Adapted from https://github.com/geyan21/ManiFlow_Policy
def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules



# Adapted from https://github.com/geyan21/ManiFlow_Policy
class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 pointwise=False,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
        
        self.pointwise = pointwise
            
         
    def forward(self, x):
        x = self.mlp(x)
        if not self.pointwise:
            x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
# Adapted from https://github.com/geyan21/ManiFlow_Policy
class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 pointwise=False,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
        
        self.pointwise = pointwise
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
        
         
    def forward(self, x):
        x = self.mlp(x)
        if not self.pointwise:
            x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class RandLANetEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels=256,
                 use_layernorm=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 num_neighbors=16, 
                 decimation=2, 
                 num_layers=4, 
                 **kwargs
                 ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.num_layers = num_layers

        cprint("randlanet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("randlanet use_final_norm: {}".format(final_norm), 'cyan')
        cprint("randlanet use_projection: {}".format(use_projection), 'cyan')

        # Build encoder layers dynamically
        encoder_layers = []
        layer_dims = [8, 16, 64, 128, 256]  # You can customize these
        # layer_dims = [8, 16, 32, 64, 128]  # You can customize these
        # layer_dims = [16, 32, 64, 128, 256]  # You can customize these
        # layer_dims = [32, 64, 128, 256, 512]  # You can customize these

        self.fc_start = nn.Linear(in_channels, layer_dims[0])
        # self.bn_start = nn.Sequential(
        #     nn.BatchNorm2d(layer_dims[0], eps=1e-6, momentum=0.99),
        #     nn.LeakyReLU(0.2)
        # )

        for i in range(min(num_layers, len(layer_dims)-1)):
            d_in_layer = layer_dims[i] if i == 0 else layer_dims[i] * 2  # *2 due to LFA output
            d_out_layer = layer_dims[i+1]
            encoder_layers.append(
                LocalFeatureAggregation(d_in_layer, d_out_layer, num_neighbors)
            )
        
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Final MLP for feature refinement
        final_dim = layer_dims[min(num_layers, len(layer_dims)-1)] * 2
        self.final_mlp = SharedMLP(final_dim, final_dim, activation_fn=nn.ReLU())
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(final_dim, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(final_dim, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
        

        # print encoder layers summary
        cprint(f"Encoder with {len(self.encoder)} layers, decimation: {self.decimation}, "
               f"final feature dimension: {final_dim}, initial input dimension: {in_channels}", 'green')
        

    def forward(self, input):
        """
        Forward pass - encoder only
        
        Parameters
        ----------  
        input: torch.Tensor, shape (B, N, d_in)
            input points
            
        Returns
        -------
        coords: torch.Tensor, shape (B, N_final, 3)
            coordinates of aggregated points
        features: torch.Tensor, shape (B, d_out, N_final)  
            aggregated features
        """
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        # x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # Random permutation for better sampling
        permutation = torch.randperm(N, device=coords.device)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        # ENCODER - Progressive decimation
        for i, lfa in enumerate(self.encoder):
            current_n = N // decimation_ratio
            x = lfa(coords[:, :current_n], x)
            
            # Decimate for next layer
            decimation_ratio *= d
            next_n = N // decimation_ratio
            x = x[:, :, :next_n]
            
            # print(f"Layer {i}: {current_n} -> {next_n} points, feature dim: {x.shape[1]}")

        # Final feature processing
        x = self.final_mlp(x)
        
        final_n = N // (d ** len(self.encoder))
        final_coords = coords[:, :final_n]  # shape (B, N_final, 3)
        final_features = x.squeeze(-1).transpose(-2, -1)  # shape (B, N_final, d_out)

        final_features = self.final_projection(final_features)  # shape (B, N_final, out_channels)
        
        return final_features

    def get_final_point_count(self, initial_points):
        """Calculate the final number of points after encoding"""
        return initial_points // (self.decimation ** len(self.encoder))

class PointNet2DenseEncoder(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=256, 
                 use_bn=False,
                 **kwargs):
        super(PointNet2DenseEncoder, self).__init__()
        
        # self.sa1 = PointNetSetAbstractionMsg(50, [0.02, 0.04, 0.08], [8, 16, 64], in_channel - 3,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(10, [0.04, 0.08, 0.16], [16, 32, 64], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        # self.sa1 = PointNetSetAbstraction(npoint=64, radius=0.04, nsample=16, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=16, radius=0.08, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.04, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False, bn=use_bn)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.08, nsample=64, in_channel=128 + 3, mlp=[128, 256, 512], group_all=False, bn=use_bn)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.08, nsample=64, in_channel=128 + 3, mlp=[128, 256, out_channels], group_all=False, bn=use_bn)

        # copy variables
        self.in_channels = in_channels
        
    def forward(self, xyz):
        # permute from B, N, C to B, C, N
        xyz = xyz.permute(0, 2, 1)  # B, N, C -> B, C, N
        B, D, N = xyz.size()
        assert D == self.in_channels
        if D > 3:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # permute back to B, N, C
        l2_points = l2_points.permute(0, 2, 1)
        return l2_points

class Uni3DPointcloudEncoder(nn.Module):
    def __init__(self, 
                pc_model='eva02_tiny_patch14_224', # eva02_tiny_patch14_224, eva02_base_patch14_448, eva02_large_patch14_448
                pc_feat_dim=192, # 192, 768, 1024
                embed_dim=1024,
                group_size=32,
                num_group=64,
                patch_dropout=0.5,
                drop_path_rate=0.2,
                pretrained_pc=None,
                pc_encoder_dim=512,
                out_channels=512,  # output dimension of the encoder
                return_cls=True,  # whether to return the cls token
                **kwargs
                #  args = {
                #     # 'pc_model': 'eva02_large_patch14_448', #eva02_tiny_patch14_224, eva02_base_patch14_448
                #     # 'pc_model': 'eva02_tiny_patch14_224',
                #     'pc_model': 'eva02_base_patch14_448',
                #     'pc_feat_dim': 768, #192, 768
                #     # 'pc_feat_dim': 192,
                #     'embed_dim': 1024, 
                #     'group_size': 32,
                #     'num_group': 512,
                #     'patch_dropout': 0.5,
                #     'drop_path_rate': 0.2,
                #     'pretrained_pc': None,
                #     'pc_encoder_dim': 512,
                #     'return_cls': False, # whether to return the cls token
                # }
    ):
        super().__init__()
        point_transformer = timm.create_model(pc_model, checkpoint_path=pretrained_pc, drop_path_rate=drop_path_rate)
        self.trans_dim = pc_feat_dim # 768
        self.embed_dim = embed_dim # 512
        self.group_size = group_size # 32
        self.num_group = num_group # 512
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim =  pc_encoder_dim # 256
        self.encoder = Encoder(encoder_channel = self.encoder_dim)

        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.return_cls = return_cls  # whether to return the cls token

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.visual_pos_drop = point_transformer.pos_drop
        self.visual_blocks = point_transformer.blocks
        self.visual_norm = point_transformer.norm
        self.visual_fc_norm = point_transformer.fc_norm
        
        weights_only=False
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        if pc_model == 'eva02_tiny_patch14_224':
            state_dict = torch.load(os.path.join(cur_dir, 'pretrained_models/uni3d-ti/model.pt'), weights_only=weights_only)['module']
        elif pc_model == 'eva02_base_patch14_448':
            state_dict = torch.load(os.path.join(cur_dir, 'pretrained_models/uni3d-b/model.pt'), weights_only=weights_only)['module']
        elif pc_model == 'eva02_large_patch14_448':
            state_dict = torch.load(os.path.join(cur_dir, 'pretrained_models/uni3d-l/model.pt'), weights_only=weights_only)['module']
        else:
            raise NotImplementedError(f"Unsupported pc_model: {pc_model}. Supported models: eva02_tiny_patch14_224, eva02_base_patch14_448, eva02_large_patch14_448")
        # state_dict = torch.load(os.path.join(cur_dir, 'pretrained_models/uni3d-l/model.pt'))['module']
        # state_dict = torch.load(os.path.join(cur_dir, 'pretrained_models/uni3d-b/model.pt'))['module']
        
        for key in list(state_dict.keys()):
            state_dict[key.replace('point_encoder.', '').replace('visual.', 'visual_')] = state_dict[key]
        for key in list(state_dict.keys()):
            if key not in self.state_dict():
                del state_dict[key]
        self.load_state_dict(state_dict)

        cprint(f"[PointcloudEncoder] Uni3D model loaded with {pc_model}, "
               f"embed_dim: {self.embed_dim}, "
               f"group_size: {self.group_size}, "
               f"num_group: {self.num_group}, "
               f"encoder_dim: {self.encoder_dim}, "
               f"return_cls: {self.return_cls}", 'green')  
        
        self.final_projection = nn.Linear(self.embed_dim, out_channels)  # output dimension of the encoder


    def forward(self, pcd):
        pcd = random_point_dropout(pcd, max_dropout_ratio=0.8)
        pts = pcd[..., :3].contiguous()
        colors = pcd[..., 3:].contiguous()
        # divide the point cloud in the same form. This is important
        _, center, features = self.group_divider(pts, colors)
        print(f"Group input shape: {features.shape}")  # B G N C
        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #  B G N
        group_input_tokens = self.encoder2trans(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        # x = x.half()
        
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)

        x = self.visual_pos_drop(x)

        # ModuleList not support forward
        for i, blk in enumerate(self.visual_blocks):
            x = blk(x)
        if self.return_cls:
            x = self.visual_norm(x[:, 0, :]).unsqueeze(1)  # return the cls token, B 1 C
        else:
            x = self.visual_norm(x[:, 1:, :]) # return the rest of the tokens, B G C
        x = self.visual_fc_norm(x)

        x = self.trans2embed(x)

        x = self.final_projection(x)  # B G out_channels
        
        # total_sum = 0

        # for param in self.parameters():
        #     total_sum += param.sum()
        
        return x
    def output_shape(self, input_shape=None):
        return [self.embed_dim]

# Adapted from https://github.com/geyan21/ManiFlow_Policy
class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 downsample_points=False,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel

        if pointcloud_encoder_cfg.get('state_mlp_size', None) is not None:
            state_mlp_size = pointcloud_encoder_cfg.state_mlp_size
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
        
        self.downsample_points = downsample_points
        if self.downsample_points:
            self.point_preprocess = point_process.fps_torch
            self.num_points = pointcloud_encoder_cfg.num_points # 4096
        else:
            self.point_preprocess = nn.Identity()
            self.num_points = self.point_cloud_shape[1]
        cprint(f"[DP3Encoder] State MLP size: {state_mlp_size}", "yellow")
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        if self.downsample_points:
            cprint(f"[DP3Encoder] Downsampling enabled. Using {self.num_points} points from the point cloud.", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "randlanet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
            else:
                pointcloud_encoder_cfg.in_channels = 3
            self.extractor = RandLANetEncoder(**pointcloud_encoder_cfg)
            cprint(f"[DP3Encoder] RandLANetEncoder with in_channels: {pointcloud_encoder_cfg.in_channels}", "red")
        elif pointnet_type == "pointnet2":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
            else:
                pointcloud_encoder_cfg.in_channels = 3
            self.extractor = PointNet2DenseEncoder(**pointcloud_encoder_cfg)
            cprint(f"[DP3Encoder] PointNet2DenseEncoder with in_channels: {pointcloud_encoder_cfg.in_channels}", "red")
        elif pointnet_type == "uni3d":
            assert use_pc_color is True, cprint(f"Uni3D encoder only supports point cloud with color, but got use_pc_color: {use_pc_color}", "red")
            self.extractor = Uni3DPointcloudEncoder(**pointcloud_encoder_cfg)
        elif pointnet_type == "identity":
            self.extractor = nn.Identity()
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

        self.pointwise = pointcloud_encoder_cfg.get('pointwise', False)
        cprint(f"[DP3Encoder] pointwise: {self.pointwise}", "red")

        cprint(f"[DP3Encoder] input points num: {self.num_points}", "red") if self.pointwise else cprint(f"[DP3Encoder] output points num: 1", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        if self.downsample_points and points.shape[1] > self.num_points:
            points, _ = self.point_preprocess(points, self.num_points)

        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel or B * N * out_channel
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64 
        if self.pointnet_type == "identity":
            return state_feat
        if len(pn_feat.shape) == 3:
            # each point has a feature
            state_feat = state_feat.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels


if __name__ == "__main__":
    import torch
    import numpy as np

    # Load the configuration
    from yacs.config import CfgNode as CN
    pointcloud_encoder_cfg = CN()
    pointcloud_encoder_cfg.in_channels = 3
    pointcloud_encoder_cfg.out_channels = 512
    pointcloud_encoder_cfg.use_layernorm = False
    pointcloud_encoder_cfg.final_norm = 'none'
    pointcloud_encoder_cfg.use_projection = True
    pointcloud_encoder_cfg.pointwise = True
    pointcloud_encoder_cfg.num_points = 1024


    state_mlp_size = (64, 64)
    state_mlp_activation_fn = nn.ReLU

    observation_space = {
        'point_cloud': (1024, 6),
        'agent_pos': (3,),
        'imagin_robot': (1024, 3)
    }
    encoder = DP3Encoder(observation_space, state_mlp_size=state_mlp_size, state_mlp_activation_fn=state_mlp_activation_fn,
                         pointcloud_encoder_cfg=pointcloud_encoder_cfg, use_pc_color=True, 
                        #  pointnet_type='pointnet',
                            # pointnet_type='randlanet',
                            # pointnet_type='pointnet2',
                            pointnet_type='uni3d',
                            downsample_points=True)

    observations = {
        'point_cloud': torch.randn(2, 1024, 6),
        'agent_pos': torch.randn(2, 3),
        'imagin_robot': torch.randn(2, 1024, 6)
    }
    output = encoder(observations)
    print(output.shape)
    print(encoder.output_shape())
    # print(output)
