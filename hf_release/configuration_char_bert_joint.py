"""HF-Hub-compatible config for Stoicheia-tagger-parser (JointModel: tagger + biaffine parser).

Self-contained (no import from configuration_char_bert.py): this ships as its own model repo,
so it duplicates the small set of CharBertEncoder architecture fields it needs rather than
depending on a sibling package being present.
"""
from transformers import PretrainedConfig


class CharBertJointConfig(PretrainedConfig):
    model_type = "char_bert_joint"

    def __init__(
        self,
        # --- CharBertEncoder backbone architecture (fine-tuned from a Stoicheia checkpoint) ---
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
        # --- fine-tune-only encoder addition: additive capitalization channel ---
        use_cap: bool = True,
        # --- tagger head config (tagger.model.TaggerConfig + tagger.edits.LabelVocab sizes) ---
        pool: str = "mean",          # "mean" | "last" word-pooling of per-character hidden states
        head_dropout: float = 0.33,
        scalar_mix: bool = True,     # ELMo-style learned softmax mix over all (depth+1) layer outputs
        xpos_len: int = 9,           # number of factored XPOS position-heads
        n_xpos_classes: tuple = (14, 4, 4, 8, 8, 5, 5, 6, 3),  # per-position class counts (read from ckpt)
        n_script: int = 18485,       # size of the lemma edit-script vocabulary (head_script)
        n_upos: int = 13,            # size of the UPOS vocabulary (head_upos)
        use_flat: bool = True,       # whether the flat full-XPOS-tag head (head_flat) is present
        n_flat_tags: int = 1249,     # size of the attested-full-XPOS-tag vocabulary (head_flat)
        # --- parser (biaffine) config (parser.biaffine.ParserConfig, read from ckpt) ---
        d_arc: int = 500,
        d_rel: int = 150,
        n_labels: int = 24,          # size of the dependency-relation (deprel) vocabulary
        parse_dropout: float = 0.33,
        # --- packing hints for the processor (not part of the architecture) ---
        max_chars: int = 2048,
        max_words: int = 384,
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

        self.pool = pool
        self.head_dropout = head_dropout
        self.scalar_mix = scalar_mix
        self.xpos_len = xpos_len
        self.n_xpos_classes = list(n_xpos_classes)
        self.n_script = n_script
        self.n_upos = n_upos
        self.use_flat = use_flat
        self.n_flat_tags = n_flat_tags

        self.d_arc = d_arc
        self.d_rel = d_rel
        self.n_labels = n_labels
        self.parse_dropout = parse_dropout

        self.max_chars = max_chars
        self.max_words = max_words
        super().__init__(**kwargs)
