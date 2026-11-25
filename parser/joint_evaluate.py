"""Load a trained JointModel (best.pt) and report ALL CoNLL-U column scores on a split:
lemma edit-script acc, UPOS acc, factored-XPOS exact-match, and UAS/LAS — the same in-house
(encodable-token) metric the joint trainer's dev eval uses.

  python -m parser.joint_evaluate --run $SYN_DATA/runs/joint_f0 --split test
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import torch

from tagger.backbone import load_backbone_auto
from tagger.conllu import read_conllu
from tagger.dataset import TaggerDataset, pack_dev_items
from tagger.edits import LabelVocab
from tagger.model import TaggerConfig

from parser.biaffine import ParserConfig
from parser.labels import DeprelVocab
from parser.joint_model import JointModel
from parser.joint_train import evaluate_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--decode", default="greedy", choices=["greedy", "mst"])
    a = ap.parse_args()
    run = Path(os.path.expandvars(a.run))
    sd = torch.load(run / "best.pt", map_location="cpu")
    cfg = sd["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = LabelVocab.load(run / "vocab.json")
    deprel_vocab = DeprelVocab(sd["deprel_vocab"])
    T, W = sd["T"], sd["W"]

    encoder, _, tokenizer = load_backbone_auto(cfg, device)
    tcfg = TaggerConfig(**sd["tcfg"])
    pcfg = ParserConfig(**sd["pcfg"])
    core = JointModel(encoder, vocab, tcfg, pcfg, W=W).to(device)
    core.load_state_dict(sd["model"])
    core.eval()

    kdir = Path(os.path.expandvars(cfg["kfold_dir"]))
    sents = list(read_conllu(kdir / f"{a.split}.conllu"))
    hf_max_len = cfg.get("hf_max_len", 512)
    ds = TaggerDataset(sents, vocab, T, W, tokenizer=tokenizer, hf_max_len=hf_max_len)
    rows, _ = pack_dev_items(ds.encs, W, tokenizer, T)

    cnt = evaluate_dev(core, core, rows, sents, deprel_vocab, cfg.get("eval_micro", 8),
                       device, T, W, a.decode, tokenizer=tokenizer)
    nw = max(int(cnt[3]), 1); na = max(int(cnt[6]), 1)
    res = dict(split=a.split, decode=a.decode, n_words=nw, n_arc=na,
               xpos_exact=round(int(cnt[0]) / nw, 4), lemma_script=round(int(cnt[1]) / nw, 4),
               upos=round(int(cnt[2]) / nw, 4), uas=round(int(cnt[4]) / na, 4),
               las=round(int(cnt[5]) / na, 4), dev_best=sd.get("dev"))
    print("JOINT TEST " + json.dumps(res), flush=True)
    with open(run / f"{a.split}_scores_{a.decode}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
