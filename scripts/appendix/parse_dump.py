"""E5 (parsing): dump per-arc predictions of a trained JointModel on a split, for error analysis.

Same loading and batching as parser/joint_evaluate.py; writes one JSON line per scored arc:
  {sent, word, form, upos, gold_head, pred_head, gold_rel, pred_rel, sent_len}
(word/heads 1-based over the encodable tokens, exactly the units the LAS/UAS metric counts).

  python scripts/appendix/parse_dump.py --run data/e0/parser_released --split test --out results/e5/parse_released_test.jsonl
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parser.biaffine import ParserConfig
from parser.joint_model import JointModel
from parser.joint_train import parse_gold
from parser.labels import DeprelVocab
from parser.model import encodable_positions
from tagger.backbone import load_backbone_auto
from tagger.conllu import read_conllu
from tagger.dataset import TaggerDataset, batch_chunk, pack_dev_items
from tagger.edits import LabelVocab
from tagger.model import TaggerConfig


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(os.path.expandvars(a.run))
    sd = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = LabelVocab.load(run / "vocab.json"); dv = DeprelVocab(sd["deprel_vocab"])
    T, W = sd["T"], sd["W"]
    encoder, _, tokenizer = load_backbone_auto(cfg, device)
    core = JointModel(encoder, vocab, TaggerConfig(**sd["tcfg"]), ParserConfig(**sd["pcfg"]), W=W).to(device)
    core.load_state_dict(sd["model"]); core.eval()
    sents = list(read_conllu(Path(os.path.expandvars(cfg["kfold_dir"])) / f"{a.split}.conllu"))
    ds = TaggerDataset(sents, vocab, T, W, tokenizer=tokenizer, hf_max_len=cfg.get("hf_max_len", 512))
    rows, _ = pack_dev_items(ds.encs, W, tokenizer, T)
    micro = cfg.get("eval_micro", 8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = las = 0
    with open(a.out, "w") as f:
        for i in range(0, len(rows), micro):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_chunk(rows[i:i + micro], T, W, tokenizer).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, arc, rel, wmask, sent_ids = core(batch)
            gh, gl, gm = parse_gold(sent_ids, sents, dv, arc.shape[1], device)
            ph, pl = core.biaffine.decode(arc, rel, wmask)
            gh, gl = gh.cpu(), gl.cpu()
            for b, si in enumerate(sent_ids):
                s = sents[si]; pos = encodable_positions(s)
                for w, p in enumerate(pos[:arc.shape[1]]):
                    if gh[b, w] == -100:
                        continue
                    t = s.tokens[p]
                    rec = dict(sent=int(si), word=w + 1, form=t.form, upos=t.upos, xpos=t.xpos, sent_len=len(pos),
                               gold_head=int(gh[b, w]), pred_head=int(ph[b, w]),
                               gold_rel=dv.rels[int(gl[b, w])], pred_rel=dv.rels[int(pl[b, w])])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1; las += rec["gold_head"] == rec["pred_head"] and rec["gold_rel"] == rec["pred_rel"]
    print(f"wrote {n} arcs, LAS {las / max(n, 1):.4f}")


if __name__ == "__main__":
    main()
