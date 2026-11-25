# 10-fold split — statistics

## Verification (independent re-check of materialized folds)

| fold | train vs val+test 8-gram hits | sentence hits | val vs test 8-gram | sentence | status |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | PASS |
| 1 | 0 | 0 | 0 | 0 | PASS |
| 2 | 0 | 0 | 0 | 0 | PASS |
| 3 | 0 | 0 | 0 | 0 | PASS |
| 4 | 0 | 0 | 0 | 0 | PASS |
| 5 | 0 | 0 | 0 | 0 | PASS |
| 6 | 0 | 0 | 0 | 0 | PASS |
| 7 | 0 | 0 | 0 | 0 | PASS |
| 8 | 0 | 0 | 0 | 0 | PASS |
| 9 | 0 | 0 | 0 | 0 | PASS |

## Per-fold output (records / Mchars emitted; Mchars excised; records dropped)

### fold 0
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 445755 | 346.5 | 307.4 | 129773 |
| train | repaired | 1481195 | 751.9 | 493.6 | 465059 |
| train | bronze | 1001393 | 1382.1 | 36.5 | 387 |
| train | inscriptions | 802664 | 106.3 | 39.6 | 233006 |
| val | pristine | 80797 | 57.5 | 24.1 | 6196 |
| val | inscriptions | 14295 | 0.0 | 0.0 | 8016 |
| test | pristine | 34250 | 80.9 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 1
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 447569 | 347.8 | 305.6 | 130472 |
| train | repaired | 1493085 | 765.6 | 504.6 | 474145 |
| train | bronze | 1000459 | 1382.1 | 36.4 | 386 |
| train | inscriptions | 802985 | 106.4 | 39.5 | 232683 |
| val | pristine | 76171 | 55.8 | 25.5 | 6934 |
| val | inscriptions | 14300 | 0.0 | 0.0 | 8011 |
| test | pristine | 37520 | 81.7 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 2
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 444220 | 350.1 | 302.9 | 128900 |
| train | repaired | 1472496 | 757.5 | 504.0 | 468375 |
| train | bronze | 1007719 | 1381.3 | 37.3 | 375 |
| train | inscriptions | 803097 | 106.4 | 39.5 | 232561 |
| val | pristine | 78685 | 53.7 | 28.3 | 10209 |
| val | inscriptions | 14287 | 0.0 | 0.0 | 8024 |
| test | pristine | 34578 | 81.5 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 3
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 434529 | 344.9 | 307.6 | 122603 |
| train | repaired | 1455808 | 737.4 | 500.0 | 459204 |
| train | bronze | 1008911 | 1381.5 | 37.1 | 368 |
| train | inscriptions | 802494 | 106.2 | 39.7 | 233212 |
| val | pristine | 94532 | 54.5 | 27.4 | 10018 |
| val | inscriptions | 14300 | 0.0 | 0.0 | 8011 |
| test | pristine | 44662 | 82.1 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 4
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 437447 | 348.9 | 304.3 | 121011 |
| train | repaired | 1484038 | 751.8 | 481.7 | 444893 |
| train | bronze | 1004991 | 1381.6 | 37.0 | 386 |
| train | inscriptions | 802484 | 106.3 | 39.6 | 233189 |
| val | pristine | 82747 | 54.9 | 26.4 | 9290 |
| val | inscriptions | 14274 | 0.0 | 0.0 | 8037 |
| test | pristine | 55498 | 82.0 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 5
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 441096 | 352.8 | 301.1 | 125439 |
| train | repaired | 1494301 | 768.5 | 483.3 | 440710 |
| train | bronze | 1006902 | 1381.1 | 37.5 | 375 |
| train | inscriptions | 803006 | 106.4 | 39.5 | 232613 |
| val | pristine | 80639 | 53.9 | 27.2 | 8154 |
| val | inscriptions | 14294 | 0.0 | 0.0 | 8017 |
| test | pristine | 42274 | 81.3 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 6
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 449599 | 354.3 | 299.5 | 123039 |
| train | repaired | 1492775 | 772.0 | 478.9 | 441769 |
| train | bronze | 1004354 | 1381.7 | 36.9 | 359 |
| train | inscriptions | 802786 | 106.4 | 39.5 | 232909 |
| val | pristine | 81238 | 54.8 | 26.5 | 9711 |
| val | inscriptions | 14288 | 0.0 | 0.0 | 8023 |
| test | pristine | 39720 | 81.3 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 7
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 447735 | 345.7 | 307.7 | 127347 |
| train | repaired | 1474914 | 753.6 | 491.4 | 458368 |
| train | bronze | 1007220 | 1381.0 | 37.6 | 368 |
| train | inscriptions | 802873 | 106.5 | 39.4 | 232836 |
| val | pristine | 87282 | 57.9 | 23.7 | 6746 |
| val | inscriptions | 14274 | 0.0 | 0.0 | 8037 |
| test | pristine | 44180 | 81.4 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 8
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 442997 | 347.3 | 305.4 | 129125 |
| train | repaired | 1464072 | 745.2 | 505.6 | 468473 |
| train | bronze | 1005029 | 1381.4 | 37.2 | 361 |
| train | inscriptions | 802790 | 106.5 | 39.4 | 232845 |
| val | pristine | 82206 | 56.2 | 25.8 | 7787 |
| val | inscriptions | 14303 | 0.0 | 0.0 | 8008 |
| test | pristine | 37405 | 81.8 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

### fold 9
| split | tier | records | Mchars | Mchars excised | dropped records |
|---|---|---|---|---|---|
| train | pristine | 443392 | 345.4 | 308.1 | 129729 |
| train | repaired | 1467398 | 740.2 | 497.6 | 466110 |
| train | bronze | 1006317 | 1381.6 | 37.0 | 388 |
| train | inscriptions | 802364 | 106.3 | 39.6 | 233302 |
| val | pristine | 79190 | 56.0 | 24.8 | 6093 |
| val | inscriptions | 14284 | 0.0 | 0.0 | 8027 |
| test | pristine | 37718 | 82.1 | 0.0 | 0 |
| test | inscriptions | 22305 | 0.0 | 0.0 | 0 |

## Adversarial double-check

- D1 mutation test (verifier detects planted verbatim AND orthographically-mutated editions; flags nothing on genuine train): **PASS**
- D2 TM/PHI digit-rule compliance + D3 id-disjointness per fold: **PASS**

## Greek-origin bronze exclusion (anti-paraphrase guard)

~102387 bronze records (back-translations of Latin translations of GREEK works: Graeca miscellanea, Biblia, Ptolemaeus Latinus, Versiones latinae, Greek-author works in PL etc.) removed from every fold's train.

## Rare-word paraphrase screen (per-fold bronze filter)

| fold | bronze dropped | Mchars |
|---|---|---|
| 0 | 1298 | 2.68 |
| 1 | 3115 | 6.64 |
| 2 | 1494 | 3.06 |
| 3 | 2579 | 5.5 |
| 4 | 1352 | 2.95 |
| 5 | 3714 | 7.41 |
| 6 | 1110 | 2.32 |
| 7 | 915 | 1.96 |
| 8 | 2495 | 5.02 |
| 9 | 2787 | 5.9 |

## Stage 1 (tokenization)

```json
{
  "pristine": {
    "records": 408219,
    "sentences": 8486437,
    "zones": {
      "-1": 388315,
      "12": 15981,
      "10": 1949,
      "11": 1974
    },
    "ddbdp_tm_fallback": 5
  },
  "repaired": {
    "records": 1743056,
    "sentences": 19561859,
    "zones": {
      "12": 1737572,
      "10": 2718,
      "11": 2766
    },
    "ddbdp_tm_fallback": 26096
  },
  "bronze": {
    "records": 676062,
    "sentences": 12728492,
    "zones": {
      "12": 676062
    },
    "ddbdp_tm_fallback": 0
  },
  "inscriptions": {
    "records": 1265818,
    "sentences": 2130160,
    "zones": {
      "12": 1012690,
      "11": 126609,
      "10": 126519
    },
    "ddbdp_tm_fallback": 0
  }
}
```

## Stage 2 (pristine clustering)

```json
{
  "verify_j": 0.7,
  "pair_min_frac": 0.5,
  "prefix_groups": 13850,
  "after_prefix": {
    "n_components": 13850,
    "top5_words": [
      707821,
      438910,
      422779,
      383873,
      362839
    ],
    "top5_sizes": [
      1,
      2,
      1,
      3,
      5305
    ]
  },
  "exact_dup_edges": 21,
  "after_exact": {
    "n_components": 13830,
    "top5_words": [
      707821,
      438910,
      422779,
      383873,
      362839
    ],
    "top5_sizes": [
      1,
      2,
      1,
      3,
      5305
    ]
  },
  "lsh_edges": 7354,
  "lsh_hot_bands_skipped": 0,
  "after_lsh": {
    "n_components": 6476,
    "top5_words": [
      100841058,
      422779,
      339597,
      335713,
      312757
    ],
    "top5_sizes": [
      364172,
      1,
      3,
      1,
      1
    ]
  },
  "sentence_edges": 1036,
  "hot_sentence_hashes_skipped": 2712,
  "after_sentences": {
    "n_components": 5440,
    "top5_words": [
      116690058,
      365589,
      302089,
      207698,
      164661
    ],
    "top5_sizes": [
      368321,
      90,
      312,
      84,
      143
    ]
  },
  "n_clusters": 5440,
  "n_literary": 388315,
  "top10_cluster_sizes": [
    368321,
    312,
    239,
    232,
    218,
    198,
    188,
    177,
    175,
    154
  ],
  "top10_cluster_words": [
    116690058,
    302089,
    6133,
    6142,
    22064,
    48085,
    5331,
    5263,
    10276,
    2781
  ],
  "largest_cluster_sample": [
    "tlg0018.tlg005.1st1K-grc1",
    "tlg0045.tlg001.1st1K-grc1",
    "tlg0046.tlg001.1st1K-grc1",
    "tlg0057.tlg003.1st1K-grc1",
    "tlg0057.tlg012.1st1K-grc1",
    "tlg0057.tlg102.1st1K-grc1",
    "tlg0059.tlg037.1st1K-grc1",
    "tlg0061.tlg004.1st1K-grc1",
    "tlg0086.tlg008.1st1K-grc1",
    "tlg0086.tlg018.1st1K-grc1",
    "tlg0086.tlg020.1st1K-grc1",
    "tlg0086.tlg024.1st1K-grc1",
    "tlg0086.tlg037.1st1K-grc1",
    "tlg0086.tlg042.1st1K-grc1",
    "tlg0086.tlg043.1st1K-grc1"
  ]
}
```

## Stage 3 (bucket packing)

```json
{
  "bucket_words": {
    "0": 13123525,
    "1": 13123525,
    "2": 13123517,
    "3": 13123517,
    "4": 13123524,
    "5": 13123524,
    "6": 13123524,
    "7": 13123524,
    "8": 13123524,
    "9": 13123524
  },
  "bucket_records": {
    "0": 32301,
    "1": 35571,
    "2": 32629,
    "3": 42713,
    "4": 53549,
    "5": 40325,
    "6": 37771,
    "7": 42231,
    "8": 35456,
    "9": 35769
  },
  "n_clusters": 5440,
  "clusters_split": 1,
  "split_units": 7668,
  "largest_cluster_words": 116690058,
  "total_words": 131235228
}
```

## Stage 3b (canonical zones)

```json
{
  "records": 4093155,
  "duplicate_rids_across_tiers": 0,
  "prefix_bucket_conflicts": 0,
  "counts": {
    "ddbdp_tm_fallback_pristine": 5,
    "ddbdp_tm_fallback_repaired": 22,
    "repaired_sibling_bucketed": 1264962,
    "zone_0_pristine": 32301,
    "zone_0_repaired": 118131,
    "zone_10_inscriptions": 126519,
    "zone_10_pristine": 1949,
    "zone_10_repaired": 2628,
    "zone_11_inscriptions": 126609,
    "zone_11_pristine": 1974,
    "zone_11_repaired": 2810,
    "zone_12_bronze": 676062,
    "zone_12_inscriptions": 1012690,
    "zone_12_pristine": 15981,
    "zone_12_repaired": 472656,
    "zone_1_pristine": 35571,
    "zone_1_repaired": 122312,
    "zone_2_pristine": 32629,
    "zone_2_repaired": 110332,
    "zone_3_pristine": 42713,
    "zone_3_repaired": 138086,
    "zone_4_pristine": 53549,
    "zone_4_repaired": 136717,
    "zone_5_pristine": 40325,
    "zone_5_repaired": 132330,
    "zone_6_pristine": 37771,
    "zone_6_repaired": 123009,
    "zone_7_pristine": 42231,
    "zone_7_repaired": 136263,
    "zone_8_pristine": 35456,
    "zone_8_repaired": 118322,
    "zone_9_pristine": 35769,
    "zone_9_repaired": 129460
  }
}
```

## Stage 4 (contamination index)

```json
{
  "raw_keys": 150282623,
  "unique_keys": 87597072,
  "contributing_records": 645366
}
```

## Stage 5 (conflict masks)

```json
{
  "pristine": {
    "records": 408219,
    "contaminated": 308274,
    "doc_lsh_hits": 29613
  },
  "repaired": {
    "records": 1743056,
    "contaminated": 1290724,
    "doc_lsh_hits": 77185
  },
  "bronze": {
    "records": 676062,
    "contaminated": 258424,
    "doc_lsh_hits": 0
  },
  "inscriptions": {
    "records": 1265818,
    "contaminated": 315070,
    "doc_lsh_hits": 0
  }
}
```
