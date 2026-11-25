"""Milestone-(a) data audit for one fold: script inventory + dev coverage, attested-tag
ceiling, lemma charset, MWT/unencodable counts.  Usage: python scripts/tagger/audit_fold.py 0"""
import os, sys, unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root (scripts/tagger/ -> ../..)
from tagger.backbone import ALPHABET
from tagger.conllu import read_conllu
from tagger.dataset import encode_word
from tagger.edits import LabelVocab, compute_script, form_key

KFOLD = Path(os.path.expandvars("$STOICHEIA_DATA/treebanks/oga_repo/kfold"))
fold = sys.argv[1] if len(sys.argv) > 1 else "0"

train = list(read_conllu(KFOLD / f"train{fold}.conllu"))
dev = list(read_conllu(KFOLD / f"dev{fold}.conllu"))
print(f"train sents={len(train)} dev sents={len(dev)}")

enc_ok = lambda f: encode_word(f) is not None
vocab = LabelVocab.build(train, enc_ok)
print(f"scripts={vocab.n_scripts} tags={len(vocab.tags)} upos={vocab.upos}")
print("xpos position alphabet sizes:", [len(a) for a in vocab.xpos_alpha])
print(f"lexicon forms={len(vocab.lex_f)} nongreek forms={len(vocab.nongreek)}")

n_tok = sum(len(s.tokens) for s in train)
n_mwt = sum(1 for s in train for k, _ in s.lines if k == "mwt")
n_empty = sum(1 for s in train for k, _ in s.lines if k == "empty")
print(f"train tokens={n_tok} mwt_ranges={n_mwt} empty_nodes={n_empty}")

# lemma charset audit (letters outside Greek+combining marks, digits, etc.)
charset = Counter()
for s in train:
    for t in s.tokens:
        if enc_ok(t.form):
            for c in unicodedata.normalize("NFD", t.lemma):
                if c.lower() not in ALPHABET and not unicodedata.category(c).startswith("M"):
                    charset[c] += 1
print("lemma non-greek chars (top 25):", charset.most_common(25))

# dev-side coverage
cov = Counter()
for s in dev:
    for t in s.tokens:
        if not enc_ok(t.form):
            cov["nongreek"] += 1
            key = t.form in vocab.nongreek
            tag = (t.xpos or "-" * 9)[:9].ljust(9, "-")
            if key and vocab.nongreek[t.form] == (t.lemma, t.upos, tag):
                cov["nongreek_rule_ok"] += 1
            continue
        cov["greek"] += 1
        key = form_key(t.form)
        sc = compute_script(key, t.lemma)
        cov["script_seen"] += vocab.script_id(sc) != -100
        tag = (t.xpos or "-" * 9)[:9].ljust(9, "-")
        cov["tag_seen"] += tag in vocab._tagset
        cov["form_in_lex"] += key in vocab.lex_f
        if key in vocab.lex_f:
            cov["lemma_in_lex"] += t.lemma in vocab.lex_f[key]

g = max(cov["greek"], 1)
print(f"dev greek tokens={cov['greek']} nongreek={cov['nongreek']} "
      f"(rule-correct {cov['nongreek_rule_ok']}/{cov['nongreek']})")
print(f"SCRIPT COVERAGE      = {cov['script_seen']/g:.4f}   (gate >= 0.98)")
print(f"ATTESTED-TAG CEILING = {cov['tag_seen']/g:.4f}")
print(f"form in lexicon      = {cov['form_in_lex']/g:.4f}")
print(f"  ...and gold lemma among its candidates = {cov['lemma_in_lex']/max(cov['form_in_lex'],1):.4f}")
