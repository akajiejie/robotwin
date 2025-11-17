import torch
import torch.nn as nn
from pytorch3d.ops import knn_points
from termcolor import cprint
from typing import Optional, Dict, Tuple, Union, List, Type

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, bn=False):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=bn, activation_fn=nn.ReLU())

    def forward(self, coords, features, knn_output):
        # Extract indices and distances from PyTorch3D KNN result
        idx = knn_output.idx  # shape (B, N, K)
        dist = knn_output.dists  # shape (B, N, K)
        
        B, N, K = idx.size()
        
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3)
        
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=bn, activation_fn=nn.ReLU())

    def forward(self, x):
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)
        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)
        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, bn=False):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, bn=bn, activation_fn=nn.LeakyReLU())
        self.mlp2 = SharedMLP(d_out, 2*d_out, bn=bn, activation_fn=nn.LeakyReLU())
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=bn)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, bn=bn)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, bn=bn)

        self.pool1 = AttentivePooling(d_out, d_out//2, bn=bn)
        self.pool2 = AttentivePooling(d_out, d_out, bn=bn)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        knn_output = knn_points(coords, coords, K=self.num_neighbors, return_nn=False)

        x = self.mlp1(features)
        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)
        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


class RandLANetEncoder(nn.Module):
    def __init__(self, 
                 d_in, 
                 num_neighbors=16, 
                 decimation=2, 
                 num_layers=4, 
                 **kwargs
                 ):
        super(RandLANetEncoder, self).__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.num_layers = num_layers

        # Build encoder layers dynamically
        encoder_layers = []
        layer_dims = [8, 16, 64, 128, 256]  # You can customize these
        # layer_dims = [8, 16, 32, 64, 128]  # You can customize these
        # layer_dims = [16, 32, 64, 128, 256]  # You can customize these
        # layer_dims = [32, 64, 128, 256, 512]  # You can customize these

        self.fc_start = nn.Linear(d_in, layer_dims[0])
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(layer_dims[0], eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )



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

        # print encoder layers summary
        cprint(f"Encoder with {len(self.encoder)} layers, decimation: {self.decimation}, "
               f"final feature dimension: {final_dim}, initial input dimension: {d_in}", 'green')
        

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
            
            print(f"Layer {i}: {current_n} -> {next_n} points, feature dim: {x.shape[1]}")

        # Final feature processing
        x = self.final_mlp(x)
        
        final_n = N // (d ** len(self.encoder))
        final_coords = coords[:, :final_n]  # shape (B, N_final, 3)
        final_features = x.squeeze(-1).transpose(-2, -1)  # shape (B, N_final, d_out)
        
        return final_coords, final_features

    def get_final_point_count(self, initial_points):
        """Calculate the final number of points after encoding"""
        return initial_points // (self.decimation ** len(self.encoder))


# Usage example
if __name__ == '__main__':
    import time
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    # Your input: 40960 points with features
    d_in = 7  # 3 coords + 4 features (or whatever you have)
    initial_points = 1024
    
    # Different configurations for different final point counts
    configs = [
        # {"decimation": 4, "num_layers": 4},  # 40960 -> 160 points
        # {"decimation": 4, "num_layers": 3},  # 40960 -> 640 points  
        {"decimation": 2, "num_layers": 4},  # 40960 -> 2560 points
        # {"decimation": 8, "num_layers": 2},  # 40960 -> 640 points
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: decimation={config['decimation']}, layers={config['num_layers']}")
        
        model = RandLANetEncoder(d_in, decimation=config['decimation'], 
                               num_layers=config['num_layers'], device=device).to(device)
        
        final_count = model.get_final_point_count(initial_points)
        print(f"Final points: {initial_points} -> {final_count}")
        
        # Test with random data
        cloud = torch.randn(1, initial_points, d_in).to(device)
        
        # Warmup
        with torch.no_grad():
            model(cloud)
        
        # Time the forward pass
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t0 = time.time()
        with torch.no_grad():
            coords, features = model(cloud)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t1 = time.time()
        
        print(f"Forward time: {(t1-t0)*1000:.2f}ms")
        print(f"Output coords shape: {coords.shape}")
        print(f"Output features shape: {features.shape}")
