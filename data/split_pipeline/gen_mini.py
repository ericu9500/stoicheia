#!/usr/bin/env python3
"""Build a miniature raw/ tree for smoke-testing the pipeline end to end.

Includes injected probes:
  P1 two pristine 'editions' of the same fake text (different orthography)
     -> must land in the same cluster/bucket
  P2 a repaired record quoting one sentence of probe text A verbatim
  P3 a bronze passage embedding the same sentence with different punctuation
  P4 a repaired record = reordered words of a probe sentence (bag match)
All of P2-P4 must be excised from train whenever probe A is in val/test.
"""
import os
import sys

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

MINI = sys.argv[1]
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE_A = ("Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος οὐλομένην. "
           "ἣ μυρί᾽ Ἀχαιοῖς ἄλγε᾽ ἔθηκε πολλὰς δ᾽ ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν. "
           "ἡρώων αὐτοὺς δὲ ἑλώρια τεῦχε κύνεσσιν οἰωνοῖσί τε πᾶσι φανερώτατον.")
PROBE_A2 = ("Μηνιν αειδε θεα Πηληιαδεω Αχιληος ουλομενην· "
            "η μυρι Αχαιοις αλγε εθηκε πολλας δ ιφθιμους ψυχας Αιδι προιαψεν· "
            "ηρωων αυτους δε ελωρια τευχε κυνεσσιν οιωνοισι τε πασι φανερωτατον.")
QUOTE = "ἣ μυρί᾽ Ἀχαιοῖς ἄλγε᾽ ἔθηκε πολλὰς δ᾽ ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν."
QUOTE_NOPUNCT = "η μυρι Αχαιοις αλγε εθηκε πολλας δ ιφθιμους ψυχας Αιδι προιαψεν"
REORDER = "Ἀχαιοῖς μυρί᾽ ἣ ἔθηκε ἄλγε᾽ πολλὰς ψυχὰς ἰφθίμους δ᾽ Ἄϊδι προΐαψεν."
FILLER = ("τοῦτο δὲ τὸ κείμενον οὐδαμοῦ ἕτερον εὑρίσκεται ἐν οὐδεμιᾷ βίβλῳ "
          "τῶν παλαιῶν συγγραφέων ὥστε καθαρὸν εἶναι παντάπασιν. ")


def sample_parquet(tier, per_source, extra_rows=()):
    import glob
    got = {}
    rows = []
    for path in sorted(glob.glob(os.path.join(
            SRC, "raw", "AncientGreek", "data", tier, "*.parquet"))):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            for r in t.to_pylist():
                s = r["source"]
                if got.get(s, 0) < per_source.get(s, per_source.get("*", 0)):
                    rows.append(r)
                    got[s] = got.get(s, 0) + 1
            if all(got.get(s, 0) >= c for s, c in per_source.items() if s != "*"):
                break
        break  # first file is enough except pristine sources spread; ok for mini
    rows.extend(extra_rows)
    d = os.path.join(MINI, "raw", "AncientGreek", "data", tier)
    os.makedirs(d, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(d, tier + "-mini.parquet"))
    print(tier, {k: v for k, v in got.items()}, "+%d injected" % len(extra_rows))


def mk(source, rid, text, tier):
    return {"source": source, "id": rid, "license": "x", "tier": tier,
            "orig_band": "p100", "clean": 1.0, "text": text}


pristine_extra = [
    mk("greek_pd", "probeA_ed1#1.0", PROBE_A + " " + FILLER * 3, "pristine"),
    mk("ia", "probeA_ed2#7", PROBE_A2 + " " + FILLER * 3, "pristine"),
]
repaired_extra = [
    mk("greek_pd", "probeQ_quote#1.1",
       FILLER * 4 + " ὡς καὶ ὁ ποιητής φησιν· " + QUOTE + " " + FILLER * 4,
       "repaired"),
    mk("ia", "probeR_reorder#2",
       FILLER * 4 + " " + REORDER + " " + FILLER * 4, "repaired"),
]

sample_parquet("pristine", {"greek_pd": 2600, "ia": 400, "oga": 200,
                            "dbbe": 200, "catholic": 200, "ddbdp": 600,
                            "dclp": 200, "sblgnt": 20, "pg": 100,
                            "gutenberg": 100}, pristine_extra)
sample_parquet("repaired", {"*": 0, "greek_pd": 2000, "ia": 1000},
               repaired_extra)

os.makedirs(os.path.join(MINI, "raw"), exist_ok=True)
with open(os.path.join(SRC, "raw", "bronze.jsonl"), "rb") as fin, \
        open(os.path.join(MINI, "raw", "bronze.jsonl"), "wb") as fout:
    for i, line in enumerate(fin):
        if i >= 3000:
            break
        fout.write(line)
    fout.write(orjson.dumps({
        "id": "99999:00000",
        "text": FILLER * 3 + " " + QUOTE_NOPUNCT + " " + FILLER * 3,
        "author": "Probe", "work": "Probe", "corpus": "probe"}) + b"\n")

d = os.path.join(MINI, "raw", "Inscriptions_2")
os.makedirs(d, exist_ok=True)
src = os.path.join(SRC, "raw", "Inscriptions_2",
                   "synthetic_editions_with_ithaca_text_fix.jsonl")
with open(src, "rb") as fin, open(os.path.join(
        d, "synthetic_editions_with_ithaca_text_fix.jsonl"), "wb") as fout:
    for i, line in enumerate(fin):
        if i >= 3000:
            break
        fout.write(line)
print("mini raw tree at", MINI)
