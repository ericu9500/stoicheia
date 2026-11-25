"""Load a trained parser (best.pt) and report UAS/LAS on the OGA test split.

  python -m parser.evaluate --run results/parser_char

NOTE: this release only supports arm="char" — see parser/train.py's module docstring for why
the "lemma"/"fused" arms (and the build_lemma_arm loader) were dropped.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import torch

from tagger.conllu import read_conllu

from parser.biaffine import ParserConfig, BiaffineHead
from parser.labels import DeprelVocab
from parser.model import SyntaxModel
from parser.train import build_char_arm, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--decode", default="greedy", choices=["greedy", "mst"])
    a = ap.parse_args()
    run = Path(os.path.expandvars(a.run))
    sd = torch.load(run / "best.pt", map_location="cpu")
    cfg = sd["cfg"]; arm = sd["arm"]
    if arm != "char":
        raise NotImplementedError(
            f'arm={arm!r} not supported in this release — only "char" is (see parser/train.py).')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deprel_vocab = DeprelVocab(sd["deprel_vocab"])

    attn = cfg.get("attn", "sdpa")
    ft = sd.get("finetune", False)
    lemma_arm = None
    char_arm, _ = build_char_arm(device, attn, finetune=ft)
    if ft:                                      # restore the fine-tuned backbone, not the pretrained one
        char_arm.model.load_state_dict(sd["char_backbone"])
    char_arm.mix.load_state_dict(sd["char_mix"])
    pcfg = ParserConfig(d_arc=cfg.get("d_arc", 500), d_rel=cfg.get("d_rel", 150),
                       dropout=0.0, n_labels=len(deprel_vocab.rels))
    head = BiaffineHead(sd["d_in"], pcfg).to(device)
    head.load_state_dict(sd["head"])
    model = SyntaxModel(arm, char_arm, lemma_arm, head).to(device)

    kdir = Path(os.path.expandvars(cfg["kfold_dir"]))
    sents = list(read_conllu(kdir / f"{a.split}.conllu"))
    T, W = cfg.get("T", 2048), cfg.get("W", 384)
    uas, las, n = evaluate(model, sents, deprel_vocab, T, W, device,
                           cfg.get("eval_micro", cfg.get("micro_batch", 16)), a.decode)
    print(f"arm={arm} split={a.split} decode={a.decode} n={n} UAS={uas:.4f} LAS={las:.4f}", flush=True)
    with open(run / f"test_scores_{a.decode}.json", "w") as f:
        json.dump(dict(arm=arm, split=a.split, decode=a.decode, n=n, uas=uas, las=las,
                      dev_best_epoch=sd["epoch"], dev_las=sd["las"]), f, indent=2)


if __name__ == "__main__":
    main()
