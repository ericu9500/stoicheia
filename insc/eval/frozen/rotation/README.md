# Rotation samples

`inscr_fold<k>_n100.json` and `pap_fold<k>_n100.json`: 1,000 gaps (100 at each length
L=1-10) per fold and domain, drawn with

    INSC_TEST_DIGIT=k INSC_VAL_DIGIT=$(( (k+1) % 10 )) python -m insc.eval.restore_strict --make-samples --split test --n 100 --samples inscr_fold<k>_n100.json
    INSC_TEST_DIGIT=k INSC_VAL_DIGIT=$(( (k+1) % 10 )) python -m insc.eval.restore_strict_papyri --make-samples --split test --n 100 --samples pap_fold<k>_n100.json

The paper's ten-fold table was computed on samples drawn by this same procedure. On these
files the released v2 rotation models give, as mean ± sd over the ten folds,
CER 16.12 ± 1.22, top-1 73.27, top-20 80.91 on inscriptions and
CER 10.21 ± 1.09, top-1 80.91, top-20 88.81 on papyri (papyri read from the public
`Ericu950/Papyri_1`).
