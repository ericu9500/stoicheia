"""Full constrained decode of a .conllu + official CoNLL-18 scoring.

  python -m tagger.evaluate --run $STOICHEIA_DATA/runs/tagger_fold0_pilot \
      --gold $STOICHEIA_DATA/treebanks/oga_repo/kfold/dev0.conllu [--no-lexicon] [--no-tag-constraint]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tagger import conll18_ud_eval as ud
from tagger.backbone import CharBertConfig, CharBertWithHidden
from tagger.conllu import read_conllu, write_conllu
from tagger.dataset import batch_rows, encode_sentence, pack_rows
from tagger.decode import LemmaDecoder, TagDecoder
from tagger.edits import XPOS_LEN, LabelVocab, form_key
from tagger.model import TaggerConfig, TaggerModel


def load_run(run_dir, device, attn="sdpa"):
    run_dir = Path(run_dir)
    sd = torch.load(run_dir / "best.pt", map_location="cpu")
    vocab = LabelVocab.load(run_dir / "vocab.json")
    p = sd["pretrain_cfg"]
    mcfg = CharBertConfig(attn_impl=attn, d_model=p["d_model"], n_heads=p["d_model"] // 64,
                          depth=p["depth"], char_window=p["char_window"],
                          qk_norm=p.get("qk_norm", True))
    model = TaggerModel(CharBertWithHidden(mcfg), vocab, TaggerConfig(**sd["tcfg"]),
                        W=sd["W"])
    model.load_state_dict(sd["model"])
    return model.to(device).eval(), vocab, sd


def rule_pred(vocab, tok):
    """Non-neural fallback (non-Greek tokens, or words lost to truncation)."""
    r = vocab.nongreek.get(tok.form)
    if r:
        return tuple(r)
    key = form_key(tok.form)
    cands = vocab.lex_f.get(key)
    lemma = max(cands, key=cands.get) if cands else tok.form
    return (lemma, vocab.fallback_upos, vocab.fallback_xpos)


@torch.no_grad()
def predict(model, vocab, sents, device, T, W, micro=16,
            use_lexicon=True, constrain_tags=True):
    tagd = TagDecoder(vocab, constrain_tags=constrain_tags)
    lemd = LemmaDecoder(vocab, use_lexicon=use_lexicon)
    # default every token to the rule path; neural predictions overwrite below
    preds = [[rule_pred(vocab, t) for t in s.tokens] for s in sents]

    encs = [encode_sentence(s) for s in sents]
    rows, truncated = pack_rows(encs, T, W)
    for i in range(0, len(rows), micro):
        chunk = rows[i:i + micro]
        batch = batch_rows(chunk, T, W)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(b)
        mask = batch["word_id"].new_zeros(len(chunk), W, dtype=torch.bool)
        for bi, rs in enumerate(batch["slots"]):
            mask[bi, :len(rs)] = True
        xp = tagd.xpos(out["xpos"], mask, out.get("flat"))
        up = tagd.upos(out["upos"], mask)
        slp = torch.log_softmax(out["script"].float(), -1)
        for bi, rs in enumerate(batch["slots"]):
            lp = slp[bi].cpu()
            for w, (si, ti) in enumerate(rs):
                tok = sents[si].tokens[ti]
                lemma = lemd(tok.form, lp[w], xpos=xp[bi][w])
                preds[si][ti] = (lemma, up[bi][w], xp[bi][w])
    return preds, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--no-lexicon", action="store_true")
    ap.add_argument("--no-tag-constraint", action="store_true")
    a = ap.parse_args()
    run = Path(os.path.expandvars(a.run))
    gold = Path(os.path.expandvars(a.gold))
    out_path = Path(a.out) if a.out else run / (gold.stem + ".pred.conllu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, sd = load_run(run, device)
    T, W = sd["T"], sd["W"]
    print(f"run={run.name} best_epoch={sd['epoch']} dev={sd.get('dev')}", flush=True)

    sents = list(read_conllu(gold))
    t0 = time.time()
    preds, truncated = predict(model, vocab, sents, device, T, W, a.micro,
                               use_lexicon=not a.no_lexicon,
                               constrain_tags=not a.no_tag_constraint)
    write_conllu(sents, preds, out_path)
    print(f"decoded {len(sents)} sents in {time.time()-t0:.0f}s "
          f"(truncated={truncated}) -> {out_path}", flush=True)

    g = ud.load_conllu_file(str(gold))
    s = ud.load_conllu_file(str(out_path))
    ev = ud.evaluate(g, s)
    res = {k: round(ev[k].f1 * 100, 2) for k in ("UPOS", "XPOS", "Lemmas") if k in ev}
    print("CONLL18  " + json.dumps(res), flush=True)

    # analysis: IV/OOV lemma + per-position XPOS accuracy (greek tokens only)
    iv = [0, 0]; oov = [0, 0]
    pos_ok = [0] * XPOS_LEN; pos_n = 0
    for s_, ps in zip(sents, preds):
        for t, (pl, pu, px) in zip(s_.tokens, ps):
            key = form_key(t.form)
            known = key in vocab.lex_f
            b = iv if known else oov
            b[0] += pl == t.lemma
            b[1] += 1
            gt = (t.xpos or "-" * XPOS_LEN)[:XPOS_LEN].ljust(XPOS_LEN, "-")
            pos_n += 1
            for p in range(XPOS_LEN):
                pos_ok[p] += px[p] == gt[p]
    print(f"lemma IV  acc={iv[0]/max(iv[1],1):.4f} (n={iv[1]})  "
          f"OOV acc={oov[0]/max(oov[1],1):.4f} (n={oov[1]})")
    print("xpos per-position acc:", [round(o / max(pos_n, 1), 4) for o in pos_ok])
    with open(run / "scores.jsonl", "a") as f:
        f.write(json.dumps(dict(gold=str(gold), lexicon=not a.no_lexicon,
                                constrain=not a.no_tag_constraint, **res)) + "\n")


if __name__ == "__main__":
    main()
