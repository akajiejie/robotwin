# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# RDT: https://github.com/thu-ml/RoboticsDiffusionTransformer
# --------------------------------------------------------

import re
import logging
from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, RmsNorm
from maniflow.model.diffusion.positional_embedding import SinusoidalPosEmb
from maniflow.model.diffusion.ditx_block import DiTXBlock, AdaptiveLayerNorm
from maniflow.model.gate.MoEgate import SparseMoeBlock
from termcolor import cprint

logger = logging.getLogger(__name__)


class FinalLayer(nn.Module):
    """DiTX最终输出层"""
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, 
            drop=0
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class GlobalFeatureMoE(nn.Module):
    """
    全局特征MoE模块
    
    在encoder和DiTX blocks之间插入，对输入特征进行全局调整
    这样设计的优势：
    1. MoE在特征进入DiTX前进行预处理，帮助模型更好地学习特征表征
    2. 全局视角处理所有特征，避免过早的模态分离
    3. 利于训练收敛，MoE专家可以学习不同的特征转换模式
    """
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        n_shared_experts: int = 1,
        moe_aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 🔥 MoE模块：对特征进行全局调整
        self.moe = SparseMoeBlock(
            embed_dim=hidden_size,
            mlp_ratio=mlp_ratio,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_shared_experts=n_shared_experts,
            aux_loss_alpha=moe_aux_loss_alpha,
        )
        
        # 时间条件自适应层归一化
        self.ada_norm = AdaptiveLayerNorm(dim=hidden_size, dim_cond=hidden_size)
        
        cprint(f"[GlobalFeatureMoE] 初始化全局特征MoE: experts={num_experts}, "
               f"top_k={num_experts_per_tok}, shared={n_shared_experts}", "cyan")
    
    def forward(self, x: torch.Tensor, time_c: torch.Tensor) -> torch.Tensor:
        """
        全局特征MoE前向传播
        
        Args:
            x: (B, L, D) 输入特征（可以是视觉+本体感知+语言的组合）
            time_c: (B, D) 时间条件嵌入
        
        Returns:
            x: (B, L, D) MoE处理后的特征
        """
        # 1. 自适应层归一化（让特征感知时间条件）
        x_normed = self.ada_norm(x, time_c)
        
        # 2. MoE全局特征调整
        x_moe = self.moe(x_normed)
        
        # 3. 残差连接
        x = x + x_moe
        
        return x
    
    def get_moe_stats(self):
        """获取MoE统计信息"""
        if hasattr(self.moe, 'moe_stats') and self.moe.moe_stats:
            return self.moe.moe_stats
        return None


class DiTXFeatureMoE(nn.Module):
    """
    DiTX模型 + 全局特征MoE架构
    
    架构设计：
    1. Encoder: 输入嵌入 + 位置编码 + 时间编码
    2. 🔥 GlobalFeatureMoE: 全局特征调整层（新增）
    3. DiTX Blocks: 标准DiTX transformer blocks
    4. Final Layer: 输出层
    
    核心改进：
    - 在encoder和DiTX blocks之间插入全局MoE模块
    - MoE对所有输入特征（context_c）进行全局预处理
    - 帮助模型训练收敛，专家学习不同的特征转换模式
    - DiTX blocks使用标准架构，不改动原有结构
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
        diffusion_target_t_embed_dim: int = 256,
        block_type: str = "DiTX",
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        pre_norm_modality: bool = False,
        language_conditioned: bool = False,
        language_model: str = "t5-small",
        # 🔥 全局特征MoE参数
        use_feature_moe: bool = True,
        feature_moe_num_experts: int = 8,
        feature_moe_experts_per_tok: int = 2,
        feature_moe_shared_experts: int = 1,
        feature_moe_aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.visual_cond_len = visual_cond_len
        self.language_conditioned = language_conditioned
        self.pre_norm_modality = pre_norm_modality
        self.use_feature_moe = use_feature_moe
        
        # Constants
        T = horizon
        self.T = T
        self.horizon = horizon

        # 输入嵌入
        self.hidden_dim = n_emb
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.vis_cond_pos_embed = nn.Parameter(
            torch.zeros(1, visual_cond_len * n_obs_steps, n_emb)
        )
        
        # 模态预归一化
        if self.pre_norm_modality:
            self.vis_norm = AdaptiveLayerNorm(dim=n_emb, dim_cond=n_emb)

        # 时间步编码器
        flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, n_emb),
        )
        flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_target_t_embed_dim),
            nn.Linear(diffusion_target_t_embed_dim, diffusion_target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_target_t_embed_dim * 4, self.hidden_dim),
        )
        self.flow_timestep_encoder = flow_timestep_encoder
        self.flow_target_t_encoder = flow_target_t_encoder
        self.timestep_target_t_adaptor = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # 语言条件编码
        if self.language_conditioned:
            self.load_T5_encoder(model_name=language_model, freeze=True)
            self.lang_adaptor = self.build_condition_adapter(
                "mlp2x_gelu", 
                in_features=self.language_encoder_out_dim, 
                out_features=n_emb
            )
            if self.pre_norm_modality:
                self.lang_norm = AdaptiveLayerNorm(dim=n_emb, dim_cond=n_emb)
        
        # 🔥 全局特征MoE（在encoder和DiTX blocks之间）
        if use_feature_moe:
            self.feature_moe = GlobalFeatureMoE(
                hidden_size=n_emb,
                mlp_ratio=mlp_ratio,
                num_experts=feature_moe_num_experts,
                num_experts_per_tok=feature_moe_experts_per_tok,
                n_shared_experts=feature_moe_shared_experts,
                moe_aux_loss_alpha=feature_moe_aux_loss_alpha,
            )
            cprint(f"[DiTXFeatureMoE] 启用全局特征MoE: experts={feature_moe_num_experts}, "
                   f"top_k={feature_moe_experts_per_tok}", "green")
            
        # Transformer blocks (使用标准DiTXBlock)
        self.block_type = block_type
        if block_type == "DiTX":
            self.blocks = nn.ModuleList([
                DiTXBlock(
                    hidden_size=n_emb, 
                    num_heads=n_head, 
                    mlp_ratio=mlp_ratio, 
                    p_drop_attn=p_drop_attn,
                    qkv_bias=qkv_bias, 
                    qk_norm=qk_norm,
                ) for _ in range(n_layer)
            ])
            cprint(f"[DiTXFeatureMoE] 初始化{n_layer}个标准DiTX块: hidden_size={n_emb}, "
                   f"num_heads={n_head}", "cyan")
        
        # 最终输出层
        self.final_layer = FinalLayer(n_emb, output_dim)

        self.initialize_weights()
        cprint(f"[DiTXFeatureMoE] 权重初始化完成", "green")
        
        logger.info(f"模型参数量: {sum(p.numel() for p in self.parameters()):,}")
    
    def build_condition_adapter(self, projector_type, in_features, out_features):
        """构建条件适配器"""
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
            raise ValueError(f'未知的projector类型: {projector_type}')

        return projector
    
    def load_T5_encoder(self, model_name, freeze=True):
        """加载T5语言编码器"""
        from transformers import T5Config, T5EncoderModel, AutoTokenizer
        T5_model_name = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        assert model_name in T5_model_name, f"模型名称{model_name}不在{T5_model_name}中"
        encoder_name = model_name
        pretrained_model_id = f"google-t5/{encoder_name}"
        encoder_cfg = T5Config()
        self.language_encoder = T5EncoderModel(encoder_cfg).from_pretrained(pretrained_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
        if freeze:
            self.language_encoder.eval()
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        self.language_encoder_out_dim = 512
        cprint(f"加载T5编码器: {encoder_name}", "green")

    def encode_text_input_T5(self, lang_cond, norm_lang_embedding=False, output_type="sentence",
                             device="cuda" if torch.cuda.is_available() else "cpu"):
        """编码文本输入"""
        language_inputs = self.tokenizer(lang_cond, return_tensors="pt", padding=True, truncation=True)
        input_ids = language_inputs["input_ids"].to(device)
        attention_mask = language_inputs["attention_mask"].to(device)
        encoder_outputs = self.language_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = encoder_outputs.last_hidden_state
        if output_type == "token":
            return token_embeddings
        sentence_embedding = torch.mean(token_embeddings, dim=1).squeeze(1)
        if norm_lang_embedding:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=-1)
        return sentence_embedding

    def initialize_weights(self):
        """权重初始化"""
        for block in self.blocks:
            # 初始化self_attn
            if hasattr(block.self_attn, 'in_proj_weight'):
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                if hasattr(block.self_attn, 'in_proj_bias') and block.self_attn.in_proj_bias is not None:
                    nn.init.zeros_(block.self_attn.in_proj_bias)
                nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
                if hasattr(block.self_attn, 'out_proj') and block.self_attn.out_proj.bias is not None:
                    nn.init.zeros_(block.self_attn.out_proj.bias)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 初始化input emb
        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0) if self.input_emb.bias is not None else None

        # 初始化pos emb
        nn.init.normal_(self.pos_emb, std=0.02)

        # 初始化timestep编码器
        for layer in self.flow_timestep_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.flow_target_t_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # 初始化visual condition embedding
        nn.init.normal_(self.vis_cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.vis_cond_obs_emb.bias, 0) if self.vis_cond_obs_emb.bias is not None else None

        # 初始化adaptor
        nn.init.normal_(self.timestep_target_t_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_target_t_adaptor.bias, 0)

        if self.language_conditioned:
            nn.init.normal_(self.lang_adaptor[0].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[0].bias, 0) if self.lang_adaptor[0].bias is not None else None
            nn.init.normal_(self.lang_adaptor[-1].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[-1].bias, 0) if self.lang_adaptor[-1].bias is not None else None
        
        if self.pre_norm_modality:
            nn.init.zeros_(self.vis_norm.cond_linear.weight)
            nn.init.constant_(self.vis_norm.cond_linear.bias[:self.hidden_dim], 1.)
            nn.init.zeros_(self.vis_norm.cond_linear.bias[self.hidden_dim:])
            if self.language_conditioned:
                nn.init.zeros_(self.lang_norm.cond_linear.weight)
                nn.init.constant_(self.lang_norm.cond_linear.bias[:self.hidden_dim], 1.)
                nn.init.zeros_(self.lang_norm.cond_linear.bias[self.hidden_dim:])

        # Zero-out final layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def get_optim_groups(self, weight_decay: float=1e-3):
        """获取优化器参数组"""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RmsNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        if self.vis_cond_pos_embed is not None:
            no_decay.add("vis_cond_pos_embed") 

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"参数{inter_params}同时在decay和no_decay中"
        assert len(param_dict.keys() - union_params) == 0, \
            f"参数{param_dict.keys() - union_params}未分配到decay或no_decay"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(self, learning_rate: float=1e-4, weight_decay: float=1e-3,
                            betas: Tuple[float, float]=(0.9,0.95)):
        """配置优化器"""
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            target_t: Union[torch.Tensor, float, int], 
            vis_cond: torch.Tensor,
            lang_cond: Union[torch.Tensor, list, str] = None,
            **kwargs):
        """
        前向传播
        Input:
            sample: (B,T,input_dim) 动作序列
            timestep: (B,) or int, 扩散时间步t
            target_t: (B,) or float, 目标时间步
            vis_cond: (B,L,vis_cond_dim) 多模态视觉条件
            lang_cond: (B,) or list, 语言条件
        Output: 
            action: (B,T,output_dim) 预测动作
        """
        # 输入嵌入
        input_emb = self.input_emb(sample)
        x = input_emb + self.pos_emb

        # 1. 时间步编码
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        timestep_embed = self.flow_timestep_encoder(timesteps)
        
        # 2. 目标时间步编码
        target_ts = target_t
        if not torch.is_tensor(target_ts): 
            target_ts = torch.tensor([target_ts], dtype=torch.float32, device=sample.device)    
        elif torch.is_tensor(target_ts) and len(target_ts.shape) == 0:
            target_ts = target_ts[None].to(sample.device)
        target_ts = target_ts.expand(sample.shape[0])
        target_t_embed = self.flow_target_t_encoder(target_ts)
        
        time_c = torch.cat([timestep_embed, target_t_embed], dim=-1)
        time_c = self.timestep_target_t_adaptor(time_c)

        # 3. 视觉条件编码
        vis_con_obs_emb = self.vis_cond_obs_emb(vis_cond)
        vis_cond_pos_embed = self.vis_cond_pos_embed[:, :vis_cond.shape[1]]
        context_c = vis_con_obs_emb + vis_cond_pos_embed
        if self.pre_norm_modality:
            context_c = self.vis_norm(context_c, time_c)

        # 4. 语言条件编码
        if self.language_conditioned:
            assert lang_cond is not None
            lang_c = self.encode_text_input_T5(lang_cond, output_type="token", device=sample.device)
            lang_c = self.lang_adaptor(lang_c)
            if self.pre_norm_modality:
                lang_c = self.lang_norm(lang_c, time_c)
            context_c = torch.cat([context_c, lang_c], dim=1)

        # 🔥 5. 全局特征MoE（在encoder和DiTX blocks之间）
        # 对所有context特征进行全局调整，帮助训练收敛
        if self.use_feature_moe:
            context_c = self.feature_moe(context_c, time_c)

        # 6. Transformer blocks (标准DiTX)
        for block in self.blocks:
            x = block(x, time_c, context_c)

        # 7. 输出层
        x = self.final_layer(x)
        x = x[:, -self.horizon:]
        
        return x
    
    def get_moe_stats(self):
        """获取MoE统计信息"""
        if self.use_feature_moe and hasattr(self.feature_moe, 'get_moe_stats'):
            return self.feature_moe.get_moe_stats()
        return None


if __name__ == "__main__":
    """测试DiTXFeatureMoE模型"""
    print("="*80)
    print("测试DiTXFeatureMoE模型 (Encoder → 全局特征MoE → 标准DiTX Blocks)")
    print("="*80)
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 参数设置
    batch_size = 2
    horizon = 10
    input_dim = 16
    output_dim = 16
    cond_dim = 256
    visual_cond_len = 128
    n_obs_steps = 2
    
    # 创建DiTXFeatureMoE模型
    model_feature_moe = DiTXFeatureMoE(
        input_dim=input_dim,
        output_dim=output_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        cond_dim=cond_dim,
        visual_cond_len=visual_cond_len,
        diffusion_timestep_embed_dim=256,
        diffusion_target_t_embed_dim=256,
        block_type="DiTX",
        n_layer=2,
        n_head=8,
        n_emb=768,
        mlp_ratio=4.0,
        p_drop_attn=0.1,
        language_conditioned=False,
        pre_norm_modality=False,
        # 全局特征MoE配置
        use_feature_moe=True,
        feature_moe_num_experts=8,
        feature_moe_experts_per_tok=2,
        feature_moe_shared_experts=1,
        feature_moe_aux_loss_alpha=0.01,
    ).to(device)
    
    # 创建输入数据
    sample = torch.randn(batch_size, horizon, input_dim).to(device)
    timestep = torch.tensor([1, 2]).to(device)
    target_t = torch.tensor([0.1, 0.2]).to(device)
    vis_cond = torch.randn(batch_size, visual_cond_len * n_obs_steps, cond_dim).to(device)
    
    print(f"\n输入形状:")
    print(f"  sample: {sample.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  target_t: {target_t.shape}")
    print(f"  vis_cond: {vis_cond.shape}")
    
    # 前向传播（训练模式）
    print(f"\n前向传播 (训练模式)...")
    model_feature_moe.train()
    output = model_feature_moe(sample, timestep, target_t, vis_cond)
    print(f"  输出形状: {output.shape}")
    assert output.shape == (batch_size, horizon, output_dim), "输出形状不匹配"
    
    # 检查MoE统计信息
    print(f"\n全局特征MoE统计信息:")
    moe_stats = model_feature_moe.get_moe_stats()
    if moe_stats:
        print(f"  - aux_loss: {moe_stats['aux_loss']:.6f}")
        print(f"  - expert_usage: {[f'{u:.3f}' for u in moe_stats['expert_usage'].tolist()]}")
        print(f"  - topk_weights_mean: {moe_stats['topk_weights_mean']:.4f}")
    
    # 参数统计
    params = sum(p.numel() for p in model_feature_moe.parameters())
    print(f"\n模型统计:")
    print(f"  总参数量: {params:,}")
    print(f"  DiTX层数: {len(model_feature_moe.blocks)}")
    print(f"  全局特征MoE: {'启用' if model_feature_moe.use_feature_moe else '禁用'}")
    
    # 梯度测试
    print(f"\n梯度测试...")
    loss = output.sum()
    loss.backward()
    print(f"  ✅ 反向传播成功")
    
    # 推理模式测试
    print(f"\n推理模式测试...")
    model_feature_moe.eval()
    with torch.no_grad():
        output_eval = model_feature_moe(sample, timestep, target_t, vis_cond)
    print(f"  输出形状: {output_eval.shape}")
    print(f"  ✅ 推理成功")
    
    print(f"\n" + "="*80)
    print("✅ 所有测试通过!")
    print("="*80)
    
    # 对比测试：有MoE vs 无MoE
    print(f"\n" + "="*80)
    print("对比测试: 全局特征MoE vs 无MoE")
    print("="*80)
    
    model_no_moe = DiTXFeatureMoE(
        input_dim=input_dim,
        output_dim=output_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        cond_dim=cond_dim,
        visual_cond_len=visual_cond_len,
        n_layer=2,
        n_head=8,
        n_emb=768,
        use_feature_moe=False,  # 关闭MoE
    ).to(device)
    
    params_with_moe = sum(p.numel() for p in model_feature_moe.parameters())
    params_no_moe = sum(p.numel() for p in model_no_moe.parameters())
    
    print(f"\n参数对比:")
    print(f"  有全局特征MoE: {params_with_moe:,} 参数")
    print(f"  无MoE (标准):   {params_no_moe:,} 参数")
    print(f"  增加:          {params_with_moe - params_no_moe:,} 参数 "
          f"(+{(params_with_moe - params_no_moe)/params_no_moe*100:.1f}%)")
    
    print(f"\n" + "="*80)

