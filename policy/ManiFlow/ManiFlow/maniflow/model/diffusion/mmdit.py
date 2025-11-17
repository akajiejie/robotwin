# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# FastDIT: https://github.com/chuanyangjin/fast-DiT
# --------------------------------------------------------

import re
import math
import logging
from typing import Union, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.vision_transformer import Mlp, RmsNorm, use_fused_attn
from maniflow.model.diffusion.positional_embedding import SinusoidalPosEmb
from maniflow.model.diffusion.mmdit_block import MMDiTBlock
from termcolor import cprint

logger = logging.getLogger(__name__)


class FinalLayer(nn.Module):
    """
    The final layer of DIT-X, adopted from RDT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x
    
class MMDiT(nn.Module):
    """
    Flow Matching model with a Transformer backbone.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 256,
        visual_cond_len: int = 1024,
        diffusion_timestep_embed_dim: int = 256,
        diffusion_stepsize_embed_dim: int = 256,
        block_type: str = "DiTX",
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        qk_rmsnorm: bool = False,
        language_conditioned: bool=False,
    ):
        super().__init__()
        # compute number of tokens for main trunk and condition encoder
        self.n_obs_steps = n_obs_steps
        self.visual_cond_len = visual_cond_len
        self.language_conditioned = language_conditioned
        
        # constants
        T = horizon
        self.T = T
        self.horizon = horizon

        # input embedding stem
        self.hidden_dim = n_emb
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb) # visual condition observation embedding
        self.vis_cond_pos_embed = nn.Parameter(
            torch.zeros(1, visual_cond_len * n_obs_steps, n_emb)
            )  # learnable visual condition positional embedding

        # timestep and stepsize cond encoder
        flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, n_emb),
        )
        flow_stepsize_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_stepsize_embed_dim),
            nn.Linear(diffusion_stepsize_embed_dim, diffusion_stepsize_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_stepsize_embed_dim * 4, self.hidden_dim),
        )
        self.flow_timestep_encoder = flow_timestep_encoder
        self.flow_stepsize_encoder = flow_stepsize_encoder
        self.timestep_stepsize_adaptor = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Language conditioning, use T5-small as default
        if self.language_conditioned:
            self.load_T5_encoder(freeze=True)
            self.lang_adaptor = self.build_condition_adapter(
                "mlp2x_gelu", 
                in_features=self.language_encoder_out_dim, 
                out_features=n_emb
            )
        
        # Building the transformer blocks
        self.block_type = block_type
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                dim_text=n_emb,
                dim_image=n_emb,
                dim_cond=n_emb,
                dim_head=n_emb // n_head,
                heads=n_head,
                qk_rmsnorm=qk_rmsnorm,
                ) for _ in range(n_layer)
        ])
        cprint(f"[MMDiT Transformer] Initialized {n_layer} MMDiT blocks with hidden size {n_emb}, "
                f"num heads {n_head}, qk_rmsnorm {qk_rmsnorm}, block type {block_type}", "green")
    
        # Final Layer
        self.final_layer = FinalLayer(n_emb, output_dim)
    
        
        # self.initialize_weights()
        # cprint(f"[DiTX Transformer] Initialized weights for DiTX", "green")
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    # language encoder
    def load_T5_encoder(self, freeze=True):
        from transformers import (
            T5Config,
            T5EncoderModel,
            AutoTokenizer
        )
        T5_model_name = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        encoder_name = T5_model_name[0]
        pretrained_model_id = f"google-t5/{encoder_name}"
        encoder_cfg = T5Config()
        self.language_encoder = T5EncoderModel(encoder_cfg).from_pretrained(
                pretrained_model_id
            )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
        if freeze:
            self.language_encoder.eval()
            # freeze the language encoder
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        self.language_encoder_out_dim = 512
        cprint(f"Loaded T5 encoder: {encoder_name}", "green")
    
    def encode_text_input_T5(self, 
                             lang_cond, 
                             norm_lang_embedding=False, 
                             device="cuda",
                             output_type="sentence"
                             ):
        language_inputs = self.tokenizer(
            lang_cond,
            return_tensors="pt",
            padding=True,
            truncation=True,
            )
        input_ids = language_inputs["input_ids"].to(device)
        attention_mask = language_inputs["attention_mask"].to(device)
        encoder_outputs = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            )
        token_embeddings = encoder_outputs.last_hidden_state
        if output_type == "token":
            return token_embeddings
        # obtain sentence embedding by averaging the token embeddings
        sentence_embedding = torch.mean(token_embeddings, dim=1).squeeze(1) # (B, 512)
        if norm_lang_embedding:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=-1)

        return sentence_embedding

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        for block in self.blocks:
            block.apply(_basic_init)
            
            # Initialize self_attn's in_proj_weight and out_proj
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)

            # Zero-out adaLN modulation layers in DiT blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Initialize input emb by normal distribution:
        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0) if self.input_emb.bias is not None else None

        # Initialize pos emb by normal distribution:
        nn.init.normal_(self.pos_emb, std=0.02)       
        # nn.init.constant_(self.pos_emb, 0) # not used 

        # Initialize diffusion step encoder:
        for layer in self.flow_timestep_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize diffusion stepsize encoder:
        for layer in self.flow_stepsize_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize conditional observation embedding:
        nn.init.normal_(self.vis_cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.vis_cond_obs_emb.bias, 0) if self.vis_cond_obs_emb.bias is not None else None

        # Initialize the adapter for timestep and stepsize
        nn.init.normal_(self.timestep_stepsize_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_stepsize_adaptor.bias, 0)

        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        
    
    def get_optim_groups(self, weight_decay: float=1e-3, group_params=False):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        if not group_params:
            cprint("Using single parameter group for optimizer, with weight decay applied to all parameters", "yellow")
            return [
                {
                    "params": list(self.parameters()),
                    "weight_decay": weight_decay,  # All parameters get the same weight decay
                }
            ]

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RmsNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        if self.vis_cond_pos_embed is not None:
            # this is a learnable parameter, so we don't want to decay it
            no_decay.add("vis_cond_pos_embed") 

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer


    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            stepsize: Union[torch.Tensor, float, int], 
            global_cond: torch.Tensor,
            lang_cond: Union[torch.Tensor, list, str] = None,
            **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, maniflow time step t
        stepsize: (B,) or float, the step size for the flow matching process
        global_cond: (B,T, vis_cond_dim) 
        lang_cond: (B,) or list of strings, language condition input
        **kwargs: additional arguments
        output: (B,T,output_dim)
        """

        """ To be implemented: residual condition, dropout, etc. """
        
        # process input
        input_emb = self.input_emb(sample) # (B, T, n_emb)
        x = input_emb + self.pos_emb # (B, T, n_emb)
 
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        timestep_embed = self.flow_timestep_encoder(timesteps) # (B, n_emb)
        # global_c = timestep_embed

        # 2. stepsize
        stepsizes = stepsize
        if not torch.is_tensor(stepsizes): 
            stepsizes = torch.tensor([stepsizes], dtype=torch.float32, device=sample.device)    
        elif torch.is_tensor(stepsizes) and len(stepsizes.shape) == 0:
            stepsizes = stepsizes[None].to(sample.device)
        stepsizes = stepsizes.expand(sample.shape[0])
        stepsize_embed = self.flow_stepsize_encoder(stepsizes) # (B, n_emb)
        global_c = torch.cat([timestep_embed, stepsize_embed], dim=-1) # (B, 2*n_emb)
        global_c = self.timestep_stepsize_adaptor(global_c) # (B, n_emb)
        
        # 3. visual condition
        vis_con_obs_emb = self.vis_cond_obs_emb(global_cond) # (B, L, n_emb)
        vis_cond_pos_embed = self.vis_cond_pos_embed[:, :global_cond.shape[1]]
        context_c = vis_con_obs_emb + vis_cond_pos_embed # (B, L, n_emb)

        # 4. language condition
        if self.language_conditioned:
            assert lang_cond is not None
            lang_c = self.encode_text_input_T5(lang_cond, output_type="token")
            lang_c = self.lang_adaptor(lang_c) # (B, L, D) or (B, D)
            context_c = torch.cat([context_c, lang_c], dim=1) # (B, L + L_lang, n_emb)

        # 5. transformer blocks
        for block in self.blocks:
            context_c, x = block(
                time_cond=global_c,
                text_tokens=context_c,
                image_tokens=x) # (B, T, n_emb)

        # 6. head
        x = self.final_layer(x)
       
        # (B, T, output_dim)
        x = x[:, -self.horizon:] # (B, T, out_channels)
        
        return x

if __name__ == "__main__":
    # Example usage of DiTX model
    torch.manual_seed(0)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(2, 10, 16).to(device)  # Batch size 2, horizon 10, input_dim 16
    timestep = torch.tensor([1, 2]).to(device)  # Example timesteps for each sample in the batch
    stepsize = torch.tensor([0.1, 0.2]).to(device)  # Example stepsizes for each sample in the batch
    vis_cond = torch.randn(2, 5, 256).to(device)  # 5 time steps of visual condition
    lang_cond = ["This is a test sentence.", "Another test sentence."]
    model = MMDiT(
        input_dim=16,
        output_dim=16,
        horizon=10,
        n_obs_steps=5,
        cond_dim=256,
        visual_cond_len=5,
        diffusion_timestep_embed_dim=256,
        diffusion_stepsize_embed_dim=256,
        block_type="MMDiT",
        n_layer=2,  # Reduced for testing
        n_head=8,
        n_emb=768,
        qk_rmsnorm=True,
        language_conditioned=True,
    )
    model = model.to(device)
    output = model(sample, timestep, stepsize, vis_cond, lang_cond)
    print("Output shape:", output.shape)  # Should be (2, 10, 768)
    assert output.shape == (2, 10, 16), "Output shape mismatch!"
    # Check if the model is initialized correctly
    logger.info("Model initialized and forward pass successful.")
    logger.info("Output shape: %s", output.shape)
