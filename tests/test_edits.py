import pytest

from tagger.edits import apply_script, compute_script, form_key, lemma_key

CASES = [
    ("ἀνθρώπου", "ἄνθρωπος"),       # accent shift + suffix change
    ("τιλλόμενος", "τίλλω"),
    ("ἠνείχετο", "ἀνέχω"),           # augment inside: degrades but must round-trip
    ("μαθητὴς", "μαθητής"),          # grave-final form (folded by form_key)
    ("δʼ", "δέ"),                    # elision with apostrophe in FORM
    ("Σωκράτης", "Σωκράτης"),        # capitalized name, identity lemma
    ("θεώ", "θεός"),
    ("οὐκ", "οὐ"),
    ("κἀγώ", "κἀγώ"),                # crasis, identity
]


@pytest.mark.parametrize("form,lemma", CASES)
def test_roundtrip(form, lemma):
    key = form_key(form)
    sc = compute_script(key, lemma)
    out = apply_script(key, sc)
    assert out == lemma_key(lemma)
    assert sc[4] == (lemma[0] != lemma[0].lower())


def test_grave_fold():
    assert form_key("μαθητὴς") == form_key("μαθητής")


def test_inapplicable():
    sc = compute_script(form_key("ἀνθρώπου"), "ἄνθρωπος")
    assert apply_script("ὁ", sc) is None or len("ὁ") >= sc[0] + sc[2]
