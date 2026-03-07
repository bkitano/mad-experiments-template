from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Try to import fla's modules (requires triton/CUDA)
try:
    from fla.layers import DeltaNet as FLADeltaNet
    from fla.modules import RMSNorm
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    FLADeltaNet = None

    # Fallback RMSNorm when fla is not available
    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization."""

        def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight


# Fallback ShortConvolution (only used when fla is not available)
class ShortConvolution(nn.Module):
    """Short 1D convolution for local context mixing (fallback implementation)."""

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,  # Depthwise convolution
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.conv(x)[:, :, :-self.conv.padding[0]]  # Causal: remove future positions
        return x.transpose(1, 2)  # (batch, seq_len, d_model)


class SwiGLU(nn.Module):
    """SwiGLU activation for channel mixing (Llama-style FFN)."""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)  # ~2.67x expansion, common for SwiGLU
        # Round to multiple of 64 for efficiency
        d_ff = ((d_ff + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (SiLU(xW1) * xW3) W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DeltaNetLayerFallback(nn.Module):
    """
    Modern DeltaNet layer following Llama-style design (pure PyTorch fallback).

    Architecture:
        Query/Key: Linear → ShortConv → SiLU → L₂Norm
        Value: Linear → ShortConv → SiLU
        Beta: Linear → Sigmoid
        Output: Delta rule(Q, K, V, β) → RMSNorm → Linear

    DeltaNet updates a key-value memory using:
        M_t = M_{t-1} + β_t * k_t ⊗ (v_t - M_{t-1}^T k_t)

    This is slower than the fla implementation but works on CPU/MPS.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1, use_short_conv: bool = True, allow_neg_eigval: bool = False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Short convolutions (optional, but recommended)
        if use_short_conv:
            self.q_conv = ShortConvolution(d_model)
            self.k_conv = ShortConvolution(d_model)
            self.v_conv = ShortConvolution(d_model)

        # Beta (learning rate for delta update)
        self.beta_proj = nn.Linear(d_model, nhead, bias=False)

        # Output normalization (RMSNorm before final projection)
        self.out_norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def _l2_normalize(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """L2 normalization."""
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) with 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Query: Linear → ShortConv → SiLU → L₂Norm
        q = self.q_proj(x)
        if self.use_short_conv:
            q = self.q_conv(q)
        q = F.silu(q)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
        q = self._l2_normalize(q, dim=-1)

        # Key: Linear → ShortConv → SiLU → L₂Norm
        k = self.k_proj(x)
        if self.use_short_conv:
            k = self.k_conv(k)
        k = F.silu(k)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self._l2_normalize(k, dim=-1)

        # Value: Linear → ShortConv → SiLU
        v = self.v_proj(x)
        if self.use_short_conv:
            v = self.v_conv(v)
        v = F.silu(v)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim)

        # Beta: Linear → Sigmoid, optionally scaled by 2 for NC¹ expressivity
        beta = torch.sigmoid(self.beta_proj(x))  # (batch, seq_len, nhead)
        if self.allow_neg_eigval:
            beta = beta * 2.

        # Delta rule update (sequential)
        outputs = []
        M = torch.zeros(batch_size, self.nhead, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            q_t = q[:, t]  # (batch, nhead, head_dim)
            k_t = k[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t].unsqueeze(-1)  # (batch, nhead, 1)

            # Read from memory: out = q^T M
            out_t = torch.einsum('bnh,bnhd->bnd', q_t, M)
            outputs.append(out_t)

            # Compute delta: δ = v - M^T k
            retrieved = torch.einsum('bnhd,bnd->bnh', M, k_t)
            delta = v_t - retrieved

            # Update memory: M = M + β * k ⊗ δ
            update = beta_t.unsqueeze(-1) * torch.einsum('bnh,bnd->bnhd', k_t, delta)

            if mask is not None:
                update = update * mask[:, t].view(batch_size, 1, 1, 1)

            M = M + update

        # Reshape output
        out = torch.stack(outputs, dim=1).reshape(batch_size, seq_len, self.d_model)

        # Output: RMSNorm → Linear
        out = self.out_norm(out)
        out = self.o_proj(out)
        out = self.dropout(out)

        return out


class DeltaNetBlock(nn.Module):
    """
    Modern DeltaNet block following Llama-style design.

    Architecture (pre-norm with residual connections):
        Token Mixing: RMSNorm → DeltaNet → Residual
        Channel Mixing: RMSNorm → SwiGLU → Residual

    Uses fla's optimized DeltaNet when available (CUDA + triton),
    otherwise falls back to pure PyTorch implementation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        use_short_conv: bool = True,  # Modern DeltaNet uses short conv
        layer_idx: int = None,
        allow_neg_eigval: bool = False,  # Multiply beta by 2 for NC¹ expressivity
    ):
        super().__init__()

        self.d_model = d_model

        # Pre-norm for token mixing
        self.attn_norm = RMSNorm(d_model)

        if HAS_FLA:
            # Use fla's optimized implementation
            # allow_neg_eigval=True multiplies beta by 2, giving range (0, 2)
            # Reference: "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"
            self.attn = FLADeltaNet(
                mode='chunk',
                d_model=d_model,
                num_heads=nhead,
                use_short_conv=use_short_conv,
                use_gate=False,
                layer_idx=layer_idx,
                allow_neg_eigval=allow_neg_eigval,
            )
            self.use_fla = True
        else:
            # Fallback to pure PyTorch
            self.attn = DeltaNetLayerFallback(d_model, nhead, dropout, use_short_conv=use_short_conv, allow_neg_eigval=allow_neg_eigval)
            self.use_fla = False

        # Pre-norm for channel mixing
        self.ffn_norm = RMSNorm(d_model)

        # SwiGLU for channel mixing (Llama-style FFN)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Token mixing: RMSNorm → DeltaNet → Residual
        if self.use_fla:
            # fla's DeltaNet returns (output, attn_weights, past_key_values)
            x = x + self.attn(self.attn_norm(x))[0]
        else:
            x = x + self.attn(self.attn_norm(x), mask)

        # Channel mixing: RMSNorm → SwiGLU → Residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class GroupDeltaNet(nn.Module):
    """
    Modern DeltaNet model for group composition/operation tasks.

    Following Llama-style architecture with DeltaNet for token mixing:
    - Token embedding (num_tokens: group elements + BOS + EOS + PAD)
    - DeltaNet blocks with SwiGLU FFN
    - Read from EOS position for classification
    - RMSNorm throughout

    Uses fla's optimized DeltaNet when available (CUDA + triton),
    otherwise falls back to pure PyTorch implementation.

    Works with any TokenSystem (S5TokenSystem, ZnTokenSystem, etc.).
    """

    def __init__(
        self,
        num_tokens: int,  # Total vocab: group elements + BOS + EOS + PAD
        num_classes: int,  # Number of output classes (group elements)
        eos_idx: int,  # Index of EOS token
        max_seq_len: int = 512,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.0,
        use_short_conv: bool = True,  # Modern DeltaNet uses short conv
        allow_neg_eigval: bool = False,  # Multiply beta by 2 for NC¹ expressivity
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.eos_idx = eos_idx
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Token embedding (123 tokens: 120 permutations + BOS/EOS/PAD)
        self.token_embed = nn.Embedding(num_tokens, d_model)
        # Positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # DeltaNet layers (uses fla when available, fallback otherwise)
        self.layers = nn.ModuleList([
            DeltaNetBlock(d_model, nhead, dropout, use_short_conv=use_short_conv, layer_idx=i, allow_neg_eigval=allow_neg_eigval)
            for i in range(num_layers)
        ])

        # Final norm before output head
        self.final_norm = RMSNorm(d_model)

        # Output head: predicts one of 120 group elements
        self.output_head = nn.Linear(d_model, num_classes, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch, max_seq_len) with format [BOS, g_1, ..., g_k, EOS, PAD, ...]
            mask: (batch, max_seq_len) attention mask (1 = attend, 0 = ignore)

        Returns:
            logits: (batch, max_seq_len, num_classes) - logits at each position
        """
        batch_size, seq_len = tokens.shape

        # Create position indices
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)

        # Embed tokens and positions
        x = self.token_embed(tokens) + self.pos_embed(positions)

        # Apply DeltaNet layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.final_norm(x)

        # Output logits at all positions
        logits = self.output_head(x)  # (batch, seq_len, num_classes)
        return logits

# Convenience factory function
def from_token_system(token_system, **kwargs) -> GroupDeltaNet:
    """Create a GroupDeltaNet from a TokenSystem instance."""
    return GroupDeltaNet(
        num_tokens=token_system.num_tokens,
        num_classes=token_system.num_classes,
        eos_idx=token_system.EOS_IDX,
        **kwargs
    )


# Backwards compatibility alias
S5DeltaNet = GroupDeltaNet


if __name__ == "__main__":
    from tasks.s5.tokens import S5TokenSystem
    from tasks.addition.tokens import ZnTokenSystem

    # Example with S5
    s5_tokens = S5TokenSystem()
    s5_model = from_token_system(s5_tokens, d_model=128, num_layers=2)
    print(f"S5 model: {s5_tokens.num_tokens} tokens, {s5_tokens.num_classes} classes")

    # Example with Z_10
    zn_tokens = ZnTokenSystem(n=10)
    zn_model = from_token_system(zn_tokens, d_model=128, num_layers=2)
    print(f"Z_10 model: {zn_tokens.num_tokens} tokens, {zn_tokens.num_classes} classes")

    # Test forward pass
    tokens = torch.randint(0, zn_tokens.num_tokens, (1, 32))
    tokens[0, 0] = zn_tokens.BOS_IDX
    tokens[0, 5] = zn_tokens.EOS_IDX
    mask = torch.ones(1, 32)
    logits = zn_model(tokens, mask)
    print(f"Output shape: {logits.shape}")