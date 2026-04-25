import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig


class RuyaAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.h   = config.num_attention_heads
        self.d   = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.drop = nn.Dropout(config.attention_dropout_prob)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        H, D = self.h, self.d
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Flash attention (PyTorch 2.0+) — VRAM tasarrufu sağlar
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=True
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class RuyaFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1  = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2  = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class RuyaBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.hidden_size)
        self.attn = RuyaAttention(config)
        self.ln2  = nn.LayerNorm(config.hidden_size)
        self.ffn  = RuyaFFN(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class RuyaGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config    = config
        self.tok_emb   = nn.Embedding(config.vocab_size, config.hidden_size,
                                       padding_idx=config.pad_token_id)
        self.pos_emb   = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop      = nn.Dropout(config.hidden_dropout_prob)
        self.blocks    = nn.ModuleList([RuyaBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f      = nn.LayerNorm(config.hidden_size)
        self.lm_head   = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    def enable_gradient_checkpointing(self, enabled=True):
        self.gradient_checkpointing = enabled

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda hidden_states: block(hidden_states, attention_mask),
                    x,
                    use_reentrant=False
                )
            else:
                x = block(x, attention_mask)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )
        return {"loss": loss, "logits": logits}

    def param_sayisi(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
