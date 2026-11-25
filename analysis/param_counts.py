"""Parameter counts for every encoder compared in the paper.
Counts total and non-embedding parameters from the public HF configs (no weights needed)."""
import json, sys
import torch
from transformers import AutoConfig, AutoModel

MODELS = {
    "GreBERTa": "bowphs/GreBerta",
    "PhilBERTa": "bowphs/PhilBerta",
    "Ancient-Greek-BERT": "pranaydeeps/Ancient-Greek-BERT",
    "XLM-R base": "xlm-roberta-base",
    "XLM-R large": "xlm-roberta-large",
    "mBERT": "bert-base-multilingual-cased",
}
MODELS.update(dict(a.split("=", 1) for a in sys.argv[1:]))
rows = []
for name, mid in MODELS.items():
    cfg = AutoConfig.from_pretrained(mid)
    with torch.device("meta"):
        m = AutoModel.from_config(cfg)
    tot = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters() if "embeddings" in n or n.startswith("shared"))
    rows.append(dict(name=name, id=mid, params_M=round(tot / 1e6, 1), non_embedding_M=round((tot - emb) / 1e6, 1),
                     vocab=getattr(cfg, "vocab_size", None), layers=getattr(cfg, "num_hidden_layers", None),
                     hidden=getattr(cfg, "hidden_size", None)))
    print(rows[-1], flush=True)
json.dump(rows, open("analysis/param_counts.json", "w"), indent=1)
