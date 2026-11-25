# Frozen evaluation samples

`strict_test_fold3_samples.json`: the 3,000 frozen samples (300 per gap length
L=1-10) of the strict protocol -- whole inscriptions, real lacunae in context,
spaces counted, PHI identifiers ending in 3 (Ithaca's own test split). Every
system in the paper's strict table reads exactly this file, so strict-table
numbers reproduce exactly from it. Text derives from the PHI corpus via the
normalization in `data/normalize.py`.
