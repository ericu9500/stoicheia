"""HF-Hub-compatible config for Stoicheia-meter (macronization + metrical scansion).

Same backbone hyperparameters as CharBertConfig (this wraps a Stoicheia backbone
fine-tuned with two extra per-letter heads), plus the two fields that change the
model's *shape* (use_cap, scalar_mix) -- head_dropout/w_mac/w_scan/class weights are
training-only and irrelevant to inference, so they aren't part of this config.
"""
from transformers import PretrainedConfig


class CharBertMeterConfig(PretrainedConfig):
    model_type = "char_bert_meter"

    def __init__(
        self,
        n_alpha: int = 24,
        mask_id: int = 24,
        blank_id: int = 25,
        pad_id: int = 26,
        n_char_ids: int = 27,
        n_boundary: int = 4,
        n_dia: int = 49,
        n_punct: int = 7,
        d_model: int = 1024,
        n_heads: int = 16,
        depth: int = 32,
        char_window: int = 256,
        attn_impl: str = "sdpa",
        qk_norm: bool = True,
        use_cap: bool = True,
        scalar_mix: bool = True,
        **kwargs,
    ):
        self.n_alpha = n_alpha
        self.mask_id = mask_id
        self.blank_id = blank_id
        self.pad_id = pad_id
        self.n_char_ids = n_char_ids
        self.n_boundary = n_boundary
        self.n_dia = n_dia
        self.n_punct = n_punct
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.char_window = char_window
        self.attn_impl = attn_impl
        self.qk_norm = qk_norm
        self.use_cap = use_cap
        self.scalar_mix = scalar_mix
        super().__init__(**kwargs)
