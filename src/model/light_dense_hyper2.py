"""
`light_dense_hyper2` — Per-second seizure detection with v2 Hybrid-A knobs.

Same per-timestep `DenseHyperedgeBlock` stack + PMA-per-t readout as
`light_dense_hyper.DenseLightHyper`, plus the v2 optional layers from
`light_dyn_hyper2`:

  * `TemporalAttentionLayer` (per-channel T×T MHA, optional pos-enc / causal)
  * `FFTMixerLayer` (per-channel AFNO-style spectral mixer over T)

Both v2 layers preserve the (B, T, N, d) tensor shape, so they slot in
either BEFORE or AFTER the hyperedge stack (`--tattn_position`). They are
disabled by default — without `--temporal_attn` / `--use_fft_mixer` the
behaviour is identical to `light_dense_hyper`.
"""
import torch
import torch.nn as nn

from model.light_dense_hyper import DenseLightHyper, DenseHyperedgeBlock
from model.light_dyn_hyper2 import TemporalAttentionLayer, FFTMixerLayer


class DenseLightHyper2(DenseLightHyper):
    """Drop-in subclass that optionally inserts the v2 mixers in the dense path."""

    def __init__(
        self,
        *args,
        temporal_attn: bool = False,
        temporal_attn_heads: int = 4,
        tattn_n_layers: int = 1,
        tattn_pos_enc: bool = False,
        tattn_position: str = "after",
        tattn_causal: bool = False,
        use_fft_mixer: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert tattn_position in ("before", "after"), tattn_position
        self.tattn_position = tattn_position
        self.use_temporal_attn = temporal_attn
        self.use_fft_mixer = use_fft_mixer

        d = self.d_model  # d_hidden after hyper_layers; d_model in/before
        if temporal_attn:
            self.temporal_attn_layers = nn.ModuleList([
                TemporalAttentionLayer(
                    d, n_heads=temporal_attn_heads,
                    use_pos_enc=tattn_pos_enc, causal=tattn_causal,
                )
                for _ in range(tattn_n_layers)
            ])
        else:
            self.temporal_attn_layers = None

        if use_fft_mixer:
            self.fft_mixer = FFTMixerLayer(d)
        else:
            self.fft_mixer = None

    def forward(self, x: torch.Tensor):
        # x: (B, N, T, d_input)
        x = self.input_norm(x)
        H_seq = self.backbone(x)                                         # (B, T, N, d)
        if self.use_node_emb:
            H_seq = H_seq + self.node_emb.unsqueeze(0).unsqueeze(0)

        # Optional v2 mixers BEFORE hyperedge stack.
        if self.tattn_position == "before":
            if self.temporal_attn_layers is not None:
                for tl in self.temporal_attn_layers:
                    H_seq = tl(H_seq)
            if self.fft_mixer is not None:
                H_seq = self.fft_mixer(H_seq)

        for layer in self.hyper_layers:
            H_seq, _ = layer(H_seq)

        # Optional v2 mixers AFTER hyperedge stack.
        if self.tattn_position == "after":
            if self.temporal_attn_layers is not None:
                for tl in self.temporal_attn_layers:
                    H_seq = tl(H_seq)
            if self.fft_mixer is not None:
                H_seq = self.fft_mixer(H_seq)

        # Per-t readout (no temporal pooling).
        z_seq = self.pma_readout_per_t(H_seq)                            # (B, T, s*d)
        logits = self.classifier(self.dropout(z_seq))                    # (B, T, n_classes)
        return logits, H_seq


class DenseLightHyper2_classification(nn.Module):
    """main.py-compatible wrapper for the v2 dense variant."""

    def __init__(self, args, num_classes, device=None,
                 backbone_type: str = "mamba",
                 static_queries: bool = False,
                 use_input_norm: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = DenseLightHyper2(
            d_input=args.input_dim,
            d_model=args.rnn_units,
            d_hidden=args.rnn_units,
            n_nodes=args.num_nodes,
            n_classes=num_classes,
            n_mamba_layers=2,
            n_hyper_layers=getattr(args, "n_hyper_layers", 2),
            n_hyperedges=getattr(args, "n_hyperedges", 8),
            n_pma_seeds=getattr(args, "n_pma_seeds", 1),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=args.dropout,
            backbone_type=backbone_type,
            static_queries=static_queries,
            use_input_norm=use_input_norm,
            use_node_emb=getattr(args, "use_node_emb", False),
            aux_type=getattr(args, "aux_type", "none"),
            # v2 knobs
            temporal_attn=getattr(args, "temporal_attn", False),
            temporal_attn_heads=getattr(args, "temporal_attn_heads", 4),
            tattn_n_layers=getattr(args, "tattn_n_layers", 1),
            tattn_pos_enc=getattr(args, "tattn_pos_enc", False),
            tattn_position=getattr(args, "tattn_position", "after"),
            tattn_causal=getattr(args, "tattn_causal", False),
            use_fft_mixer=getattr(args, "use_fft_mixer", False),
        )

    def forward(self, input_seq, seq_lengths, adj_unused=None):
        x = input_seq.permute(0, 2, 1, 3).contiguous()
        logits, hidden = self.model(x)
        return logits, hidden

    def compute_aux_loss(self, y):
        return self.model.compute_aux_loss(y)
