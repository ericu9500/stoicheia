"""Tests for hf_release/processing_char_bert_joint.py's to_ud(): converts the tagger-
parser's native Perseus/AGDT output (Prague-style deprels, single-letter XPOS/UPOS) to
Universal Dependencies. Covers the XPOS->UPOS+FEATS table and the three restructuring
rules (COORD, AuxP, Pnom), including the promoted-node fallback deprel ("dep"/"obl")
established by direct inspection of real training-corpus examples (oga_sota.conllu):
COORD's and AuxP's own outbound deprel is always the literal structural label itself
("COORD"/"AuxP"), never a recoverable functional role, so the promoted first-conjunct/
nominal cannot inherit a meaningful relation and gets UD's generic fallback instead."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hf_release"))

from processing_char_bert_joint import to_ud, xpos_to_upos_feats


def _by_form(words):
    return {w["form"]: w for w in words}


def test_xpos_to_upos_feats_verb_and_noun():
    upos, feats = xpos_to_upos_feats("v3spia---")
    assert upos == "VERB"
    assert feats == {"Person": "3", "Number": "Sing", "Tense": "Pres",
                      "Mood": "Ind", "VerbForm": "Fin", "Voice": "Act"}

    upos, feats = xpos_to_upos_feats("n-s----n-")
    assert upos == "NOUN"
    assert feats == {"Number": "Sing", "Case": "Nom"}


def test_simple_sentence_sbj_pred_atr():
    sent = [
        dict(form="ὁ", xpos="l-s----n-", lemma="ὁ", upos="l", head=2, deprel="ATR"),
        dict(form="ἀνὴρ", xpos="n-s----n-", lemma="ἀνήρ", upos="n", head=3, deprel="SBJ"),
        dict(form="λέγει", xpos="v3spia---", lemma="λέγω", upos="v", head=0, deprel="PRED"),
    ]
    out = _by_form(to_ud(sent))
    assert out["ὁ"]["deprel"] == "det" and out["ὁ"]["upos"] == "DET"
    assert out["ἀνὴρ"]["deprel"] == "nsubj"
    assert out["λέγει"]["deprel"] == "root" and out["λέγει"]["head"] == 0


def test_coord_promotes_first_conjunct_with_generic_fallback():
    # "Παῦλος" is COORD's only child in this construction, so it gets promoted to
    # COORD's external attachment; COORD's own deprel carries no recoverable function,
    # so the fallback is "dep" (not the pre-fix bug of leaking "COORD" through verbatim).
    sent = [
        dict(form="Πέτρος", xpos="n-s----n-", lemma="Πέτρος", upos="n", head=4, deprel="SBJ"),
        dict(form="καὶ", xpos="c--------", lemma="καί", upos="c", head=1, deprel="COORD"),
        dict(form="Παῦλος", xpos="n-s----n-", lemma="Παῦλος", upos="n", head=2, deprel="SBJ"),
        dict(form="ἔρχονται", xpos="v3ppie---", lemma="ἔρχομαι", upos="v", head=0, deprel="PRED"),
    ]
    out = _by_form(to_ud(sent))
    assert out["καὶ"]["deprel"] == "cc" and out["καὶ"]["head"] == 3  # cc attaches to promoted conjunct
    assert out["Παῦλος"]["deprel"] == "dep" and out["Παῦλος"]["head"] == 1
    assert out["ἔρχονται"]["deprel"] == "root"


def test_auxp_promotes_nominal_to_obl():
    # "ἐν τῇ πόλει" (in the city): AuxP's own deprel is always the literal "AuxP",
    # never a recoverable function, so the promoted nominal falls back to "obl".
    sent = [
        dict(form="λέγει", xpos="v3spia---", lemma="λέγω", upos="v", head=0, deprel="PRED"),
        dict(form="ἐν", xpos="r--------", lemma="ἐν", upos="r", head=1, deprel="AuxP"),
        dict(form="τῇ", xpos="l-s----d-", lemma="ὁ", upos="l", head=3, deprel="ATR"),
        dict(form="πόλει", xpos="n-s----d-", lemma="πόλις", upos="n", head=2, deprel="ADV"),
    ]
    out = _by_form(to_ud(sent))
    assert out["πόλει"]["deprel"] == "obl" and out["πόλει"]["head"] == 1  # promoted, attaches to verb
    assert out["ἐν"]["deprel"] == "case" and out["ἐν"]["head"] == 4      # preposition demoted under nominal
    assert out["τῇ"]["deprel"] == "det" and out["τῇ"]["head"] == 3       # article still modifies the noun


def test_pnom_copula_reversal():
    # "ὁ ἀνὴρ ἐστι σοφός" (the man is wise): nominal becomes the head, copula
    # becomes `cop`, subject reattaches from the verb to the nominal.
    sent = [
        dict(form="ὁ", xpos="l-s----n-", lemma="ὁ", upos="l", head=2, deprel="ATR"),
        dict(form="ἀνὴρ", xpos="n-s----n-", lemma="ἀνήρ", upos="n", head=3, deprel="SBJ"),
        dict(form="ἐστι", xpos="v3spia---", lemma="εἰμί", upos="v", head=0, deprel="PRED"),
        dict(form="σοφός", xpos="a-s----n-", lemma="σοφός", upos="a", head=3, deprel="PNOM"),
    ]
    out = _by_form(to_ud(sent))
    assert out["σοφός"]["deprel"] == "root" and out["σοφός"]["head"] == 0
    assert out["ἐστι"]["deprel"] == "cop" and out["ἐστι"]["head"] == 4
    assert out["ἀνὴρ"]["deprel"] == "nsubj" and out["ἀνὴρ"]["head"] == 4  # reattached off the verb


def test_empty_sentence():
    assert to_ud([]) == []
