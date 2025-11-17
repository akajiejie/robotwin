import math
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.vision_transformer import Mlp
import numpy as np

def logit_normal_sampler(m, s=1, beta_m=100, sample_num=1000000):
    """
    Sampler from the logit-normal distribution.
    """
    y_samples = torch.randn(sample_num) * s + m
    x_samples = beta_m * (torch.exp(y_samples) / (1 + torch.exp(y_samples)))
    return x_samples


def mu_t(t, a=5, mu_max=4):
    """
    The mu(t) function
    """
    t = t.to('cpu')
    return 2 * mu_max * t ** a - mu_max


def get_beta_s(t, a=5, beta_m=100):
    """
    Get the beta_s for the logit-normal distribution.
    """
    mu = mu_t(t, a=a)
    sigma_s = logit_normal_sampler(m=mu, beta_m=beta_m, sample_num=t.shape[0])
    return sigma_s


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """
    Get 1D positional embedding in the form of sin and cos.
    
    Paper:
    https://arxiv.org/abs/1706.03762
    
    Source:
    https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
    
    Args:
        embed_dim (int): output dimension for each position.
        pos (ndarray | list): a list of positions to be encoded, size (M,).
    Returns:
        out (ndarray): resulting positional embedding, size (M, D).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim: int, grid_sizes):
    """
    Get ND positional embedding from grid sizes.
    All dimensions are summed up for factorization.
    
    Paper:
    https://arxiv.org/abs/2307.06304
    
    Args:
        embed_dim (int): output dimension for each position.
        grid_sizes (tuple): grids sizes in each dimension, length = K.
            If some grid size is lower than 1, we do not add any positional embedding.
    Returns:
        out (ndarray): resulting positional embedding, size (grid_sizes[0], ..., grid_sizes[K-1], D).
    """
    # We sum up all dimensions for factorization
    emb = np.zeros(grid_sizes + (embed_dim,))
    for size_idx, grid_size in enumerate(grid_sizes):
        # For grid size of 1, we do not need to add any positional embedding
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [embed_dim]
        posemb_shape[size_idx] = -1
        emb += get_1d_sincos_pos_embed_from_grid(embed_dim, pos).reshape(posemb_shape)
    return emb

def get_multimodal_pos_embed(embed_dim: int, mm_lens: OrderedDict):
    """
    Generate position embeddings for multimodal inputs. 
    
    Args:
        mm_lens (OrderedDict): an OrderedDict containing 
            (modality name, modality token length) pairs.
            For `"image"` modality, the value can be a multi-dimensional tuple.
            If the length < 0, it means there is no position embedding for the modality or grid.
    Returns:
        out (ndarray): positional embeddings for multimodal inputs, size (seq_len, embed_dim).
    """
    # Get total length
    tot_len = 0
    for modality, cond_len in mm_lens.items():
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            tot_len += np.prod([abs(x) for x in cond_len])
        else:
            tot_len += abs(cond_len)
    
    num_modalities = len(mm_lens)
    if num_modalities > 1:
        # Embed modality information
        modality_pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim, np.arange(num_modalities))
        
    # Get embeddings for positions inside each modality
    pos_emb = np.zeros((tot_len, embed_dim))
    start_pos = 0
    for idx, (modality, cond_len) in enumerate(mm_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            pos_embed_i_ = get_nd_sincos_pos_embed_from_grid(
                embed_dim, embed_grid_sizes)
            pos_embed_i = np.zeros(all_grid_sizes + (embed_dim,))
            pos_embed_i += pos_embed_i_
            pos_embed_i = pos_embed_i.reshape((-1, embed_dim))
        else:
            pos_embed_i_ = get_1d_sincos_pos_embed_from_grid(
                embed_dim, np.arange(cond_len)) if cond_len > 1 else 0
            pos_embed_i = np.zeros((abs(cond_len), embed_dim))
            pos_embed_i += pos_embed_i_
        
        if num_modalities > 1:
            pos_embed_i += modality_pos_embed[idx]
        # Aggregate the positional embeddings
        pos_emb[start_pos:start_pos + len(pos_embed_i)] = pos_embed_i
        start_pos += len(pos_embed_i)
    
    return pos_emb


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) module.
    
    Paper:
    https://arxiv.org/abs/1910.07467
    
    Source:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    A self-attention layer with flash attention.
    
    Paper:
    https://arxiv.org/abs/1706.03762
    
    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_heads = config["num_heads"]
        self.n_kv_heads = self.n_heads if config["num_kv_heads"] is None else config["num_kv_heads"]
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError('num_heads should be divisible by num_kv_heads')
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = config["hidden_size"]
        if self.hidden_size % self.n_heads != 0:
            raise ValueError('hidden_size should be divisible by num_heads')
        self.head_size = self.hidden_size // self.n_heads

        self.wq = nn.Linear(
            self.hidden_size, 
            self.n_heads * self.head_size, 
            bias=False
        )
        self.wkv = nn.Linear(
            self.hidden_size, 
            self.n_kv_heads * self.head_size * 2, 
            bias=False
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_size, 
            self.hidden_size, 
            bias=False
        )

        self.norm_eps = config["norm_eps"]
        self.norm_q = RMSNorm(self.head_size, eps=self.norm_eps)
        self.norm_k = RMSNorm(self.head_size, eps=self.norm_eps)

        self.use_flash_attn = config["use_flash_attn"]

        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(
        self,
        x: torch.Tensor
    ):
        bs, seq_len, _ = x.shape   # (bs, seq_len, hidden_size), batch size, sequence length, hidden size

        xq = self.wq(x)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_size)

        xkv = self.wkv(x)
        xkv = xkv.view(bs, seq_len, self.n_kv_heads, self.head_size, 2)
        xk, xv = xkv.unbind(-1)

        xq, xk = self.norm_q(xq), self.norm_k(xk)

        # Repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(
            xk, self.n_rep
        )  # (bs, seq_len, n_heads, head_size)
        xv = repeat_kv(
            xv, self.n_rep
        )  # (bs, seq_len, n_heads, head_size)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_size)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.use_flash_attn:
            output = F.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)   # (bs, n_heads, seq_len, head_size)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)


class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    
    Paper:
    https://arxiv.org/abs/1706.03762
    
    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_heads = config["num_heads"]
        self.n_kv_heads = self.n_heads if config["num_kv_heads"] is None else config["num_kv_heads"]
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError('num_heads should be divisible by num_kv_heads')
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = config["hidden_size"]
        if self.hidden_size % self.n_heads != 0:
            raise ValueError('hidden_size should be divisible by num_heads')
        self.head_size = self.hidden_size // self.n_heads

        self.wq = nn.Linear(
            self.hidden_size, 
            self.n_heads * self.head_size, 
            bias=False
        )
        self.wkv = nn.Linear(
            self.hidden_size, 
            self.n_kv_heads * self.head_size * 2, 
            bias=False
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_size, 
            self.hidden_size, 
            bias=False
        )

        self.norm_eps = config["norm_eps"]
        self.norm_q = RMSNorm(self.head_size, eps=self.norm_eps)
        self.norm_k = RMSNorm(self.head_size, eps=self.norm_eps)

        self.use_flash_attn = config["use_flash_attn"]

        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bs, seq_len, _ = x.shape   # (bs, seq_len, hidden_size), batch size, sequence length, hidden size
        _, c_len, _ = c.shape     # (bs, c_len, hidden_size), batch size, condition length, hidden size

        xq = self.wq(x)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_size)

        ckv = self.wkv(c)
        ckv = ckv.view(bs, c_len, self.n_kv_heads, self.head_size, 2)
        ck, cv = ckv.unbind(-1)

        xq, ck = self.norm_q(xq), self.norm_k(ck)

        # Repeat k/v heads if n_kv_heads < n_heads
        ck = repeat_kv(
            ck, self.n_rep
        )  # (bs, c_len, n_heads, head_size)
        cv = repeat_kv(
            cv, self.n_rep
        )  # (bs, c_len, n_heads, head_size)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_size)
        ck = ck.transpose(1, 2)  # (bs, n_heads, c_len, head_size)
        cv = cv.transpose(1, 2)  # (bs, n_heads, c_len, head_size)

        # Prepare attn mask (bs, c_len) to mask the condition
        if mask is not None:
            mask = mask.reshape(bs, 1, 1, c_len)
            mask = mask.expand(-1, -1, seq_len, -1)

        if self.use_flash_attn:
            output = F.scaled_dot_product_attention(
                query=xq,
                key=ck,
                value=cv,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(xq, ck.transpose(2, 3)) * self.attn_scale
            if mask is not None:
                scores = scores.masked_fill_(mask.logical_not(), float('-inf'))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, cv)   # (bs, n_heads, seq_len, head_size)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)
    

######## HRDT Blocks and Action Decoder ########

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    Source:
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForward(nn.Module):
    """
    A feed-forward network with SiLU activation.

    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # Apply custom dimension factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HRDTBlock(nn.Module):
    """
    H-RDT block with self-attention, two cross-attention layers and feed-forward network
    Training mode controls which cross-attention layers to use:
    - 'lang': image + language cross-attention
    """
    def __init__(self, layer_idx: int, config: dict, training_mode: str = 'lang'):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.norm_eps = config["norm_eps"]
        self.training_mode = training_mode
        
        # Validate training mode
        if training_mode not in ['lang']:
            raise ValueError(f"training_mode must be 'lang', got {training_mode}")
        
        # Self-attention layer
        self.attn_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.attn = Attention(config)
        
        # Image cross-attention layer (always present)
        self.img_cross_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.img_cond_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.img_cross_attn = CrossAttention(config)
        
        # Language cross-attention layer
        self.lang_cross_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.lang_cond_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.lang_cross_attn = CrossAttention(config)
        
        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.ffn = FeedForward(
            dim=self.hidden_size,
            hidden_dim=4*self.hidden_size,
            multiple_of=config["multiple_of"],
            ffn_dim_multiplier=config["ffn_dim_multiplier"],
        )
        
        # AdaLN modulation - keep original 9 parameters structure
        # self_attn(3) + cross_attn(3) + mlp(3) = 9 total
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 9*self.hidden_size, bias=True)
        )
        
    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cross_contexts: dict = None,
        ):
        """
        Forward pass with two cross-attention layers based on training mode
        
        Args:
            x: Input state-action sequence
            t: Timestep embedding (no sentence token anymore)
            cross_contexts: Dictionary containing cross-attention contexts
                - 'img_c': Image features for cross-attention (always used)
                - 'lang_c': Language tokens for cross-attention (if training_mode='lang')
                - 'lang_attn_mask': Attention mask for language
        """
        if cross_contexts is None:
            cross_contexts = {}
            
        # Adaptive Layer Normalization - split into shifts, scales and gates
        shift_attn, scale_attn, gate_attn, \
        shift_cross, scale_cross, gate_cross, \
        shift_mlp, scale_mlp, gate_mlp \
            = self.adaLN_modulation(t).chunk(9, dim=1)
            
        # Self-attention
        h = x + gate_attn.unsqueeze(1) * self.attn(
            modulate(self.attn_norm(x), shift_attn, scale_attn))
        
        # Image cross-attention (always present)
        img_c = cross_contexts.get('img_c')
        if img_c is not None:
            h = h + gate_cross.unsqueeze(1) * self.img_cross_attn(
                modulate(self.img_cross_norm(h), shift_cross, scale_cross),
                self.img_cond_norm(img_c), None)
        
        # Language cross-attention
        lang_c = cross_contexts.get('lang_c')
        lang_attn_mask = cross_contexts.get('lang_attn_mask')
        if lang_c is not None:
            # Apply additional cross-attention for language using same modulation parameters
            h = h + self.lang_cross_attn(
                self.lang_cross_norm(h),
                self.lang_cond_norm(lang_c), lang_attn_mask)
        
        # Feedforward network
        out = h + gate_mlp.unsqueeze(1) * self.ffn(
            modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        
        return out


class ActionDecoder(nn.Module):
    """
    The action decoder layer of H-RDT (previously called FinalLayer).
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.norm_eps = config["norm_eps"]
        self.output_size = config["output_size"]

        self.ffn_norm = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps)
        self.ffn = Mlp(
            in_features=self.hidden_size,
            hidden_features=self.hidden_size*4,
            out_features=self.output_size,
            act_layer=nn.SiLU, drop=0.0
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2*self.hidden_size, bias=True)
        )

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor
        ):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.ffn_norm(x), shift, scale)
        x = self.ffn(x)
        return x


# Keep FinalLayer for backward compatibility
FinalLayer = ActionDecoder

if __name__ == "__main__":
    # Test HRDTBlock
    config = {
        "hidden_size": 512,
        "num_heads": 8,
        "num_kv_heads": 8,
        "norm_eps": 1e-6,
        "multiple_of": 256,
        "ffn_dim_multiplier": 4.0,
        "use_flash_attn": True
    }
    hrdt_block = HRDTBlock(layer_idx=0, config=config, training_mode='lang')
    x = torch.randn(2, 10, config["hidden_size"])  # (batch_size, seq_len, hidden_size)
    t = torch.randn(2, config["hidden_size"])  # (batch_size, hidden_size)
    cross_contexts = {
        'img_c': torch.randn(2, 20, config["hidden_size"]),  # Image features
        'lang_c': torch.randn(2, 15, config["hidden_size"]),  # Language tokens
        # 'lang_attn_mask': torch.randn(2, 1, 10, 15)  # Attention mask for language
    }
    
    output = hrdt_block(x, t, cross_contexts)
    print("Output shape:", output.shape)
