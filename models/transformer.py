import torch
import torch.nn as nn

class GroupTransformer(nn.Module):
    """
    Transformer for S_5 composition using single-token representation.

    Following Grazzi et al. (ICLR 2025) with BOS/EOS tokens:
    - Token embedding (123 tokens: 120 permutations + BOS + EOS + PAD)
    - Learnable positional embeddings
    - Transformer encoder layers
    - Read from EOS position for classification
    - Output head: d_model -> 120 classes (group elements only)

    Input: (batch, max_seq_len) with tokens [BOS, g_1, ..., g_k, EOS, PAD, ...]
    Output: (batch, 120) logits for the composed permutation
    """

    def __init__(
        self,
        num_tokens: int = 123,  # 120 group elements + BOS + EOS + PAD
        num_classes: int = 120,  # Output is always a group element
        max_seq_len: int = 512,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Token embedding (123 tokens: 120 permutations + BOS/EOS/PAD)
        self.token_embed = nn.Embedding(num_tokens, d_model)
        # Positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: predicts one of 120 group elements
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

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

        # Create attention mask (True = ignore in PyTorch)
        attn_mask = (mask == 0)

        # Transform
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Output logits at all positions
        logits = self.output_head(x)  # (batch, seq_len, num_classes)
        return logits

