# Test-split manifests

One row per test record in `fold_k/test_manifest.tsv`; aggregated
by work/volume/document in `fold_k/test_works.tsv`.
Every text listed is verified to share no 8-word sequence and no
complete sentence with that fold's train and val sets (after
orthographic normalization) — safe targets for text-reconstruction
evaluation of a model trained on the same fold's train set.

## fold 0 — 56555 records, 25665 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 32301 | 80.1 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `oga:tlg5026.tlg001` (1 recs, 4.36 Mchars) Scholia in Homerum — Scholia in Iliadem
- `oga:tlg0525.tlg001` (1 recs, 1.35 Mchars) Pausanias — Description of Greece
- `ia:operaomniaquaeex09johnuoft` (135 recs, 1.18 Mchars) ἐπίδευε, τὰ λυκόδηκτα ἰῷ, τοὺς λύκους ἀπόστρεφε, τοὺς κύνας 
- `ia:pg49chryso3` (64 recs, 1.10 Mchars) Θεοῦ, σῶσον σεαυτόν αὐτὸς δὲ ἐσπούδαζεν ὅπως ἐκείνους σώσῃ τ
- `oga:tlg4036.tlg001` (1 recs, 1.00 Mchars) Proclus — In Platonis Rem Publicam Commentarii
- `oga:tlg0732.tlg005` (1 recs, 0.93 Mchars) Alexander of Aphrodisias — In Aristotelis Analyticorum Priorum Librum I Commentarium
- `ia:operaomniaquaeex05johnuoft` (107 recs, 0.91 Mchars) ἀπετέλεσε· τούτῳ πυρπολούμενος ὁ καλλίνικος Στέφανος, τὴν Ἰο
- `oga:tlg4015.tlg004` (1 recs, 0.82 Mchars) John Philoponus — In Aristotelis Analytica Posteriora Commentaria
- `ia:opera00pusegoog` (52 recs, 0.79 Mchars) μαρτύρατο "Οὐδὲν γὰρ ἐμαυτῷ σύνοιδα" διαῤῥήδην ἀναβοᾷ.λεγέτω
- `oga:tlg1799.tlg001` (1 recs, 0.76 Mchars) Euclid — Elementa
- `greek_pd:aristotelisoper01arisuoft` (551 recs, 0.70 Mchars) νητον καὶ ἀγένητον καὶ ἀναλλοιωτον καὶ εἰ πλείους εἶεν αἱ κύ
- `ia:on-the-procession-of-the-holy-spirit-original-documents-only` (42 recs, 0.68 Mchars) προσθήκη εἰς ἴσον ἔρχεται ὅπερ τῇ δευτέρᾳ συνόδῳ πέπρακται π

## fold 1 — 59825 records, 25634 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 35571 | 80.9 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `oga:tlg3135.tlg001` (2 recs, 2.84 Mchars) Joannes Zonaras — Epitome Historiarum
- `oga:tlg4013.tlg001` (1 recs, 1.55 Mchars) Simplicius — In Aristotelis Quattuor Libros De Caelo Commentaria
- `oga:pta0005.pta001` (2 recs, 1.35 Mchars) Cyrill of Alexandria — De adoratione et cultu in spiritu et ueritate
- `ia:bub_gb_T8b5d1_W00IC` (62 recs, 1.00 Mchars) ις. Γυνὴ εἰς δικαστήριον οὐκ εἰσέρχεται, οὐδὲ παῤρησίαν ἔχει
- `oga:tlg2703.tlg001` (1 recs, 0.97 Mchars) Anna Comnena — Alexias
- `ia:concordancetogre00moul` (63 recs, 0.96 Mchars) Ἀ ἦσαν δὲ οἱ πάντες ἄνδρες ὡσεὶ δώδεκα ὡσεὶ ἐπὶ ὥρας δύο κρα
- `oga:tlg0363.tlg001` (1 recs, 0.95 Mchars) Claudius Ptolemaeus — Syntaxis mathematica
- `oga:tlg4102.tlg038` (1 recs, 0.94 Mchars) Catenae (Novum Testamentum) — Catena In Epistulam Ad Hebraeos (E Cod. Paris. Coislin. 204)
- `ia:patrologiaecurs39hopfgoog` (85 recs, 0.81 Mchars) κοῦ θανάτου σκιαγραφία τις ὑπάρχει· καὶ τύπος Εἰ γὰρ σύμφυτο
- `ia:ConstPorphExcerptaHistorica1.1DeBoor` (49 recs, 0.80 Mchars) λαμβάνετε, πόλεμος ἀπολώλεκε, θεὸς δὲ ἴσως καὶ τὸ Ῥωμαίων σέ
- `ia:patrologiae_cursus_completus_gr_vol_062` (43 recs, 0.73 Mchars) εἰς τὰς ἀναγκαίας χρείας, ἵνα μὴ ὦσιν ἄκαρποι. Ἀσαδίζονταί σ
- `oga:tlg4019.tlg003` (1 recs, 0.66 Mchars) Olympiodorus — In Aristotelis Meteora Commentaria

## fold 2 — 56883 records, 25680 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 32629 | 80.7 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `oga:tlg4013.tlg004` (1 recs, 2.45 Mchars) Simplicius — In Aristotelis Physicorum Libros Commentaria
- `oga:tlg0008.tlg001` (1 recs, 1.70 Mchars) Athenaeus of Naucratis — Deipnosophistae
- `oga:tlg2021.tlg002` (1 recs, 1.18 Mchars) Epiphanius — Panarion (Adversus Haereses)
- `oga:tlg4018.tlg001` (1 recs, 1.03 Mchars) Asclepius — In Aristotelis Metaphysicorum Libros A-Z Commentaria
- `ia:pg52chryso6` (57 recs, 0.99 Mchars) θάνατον ἐκφυγεῖν, ὡς τὸ καταφρονεῖν θανάτου. Ἵνα δὲ μηδεὶς ν
- `oga:tlg0057.tlg066` (1 recs, 0.98 Mchars) Galen — De methodo medendi
- `oga:tlg2040.tlg004` (1 recs, 0.87 Mchars) Basil, Saint, Bishop of Caesarea — Epistulae
- `ia:operaomniaquaeex08john` (94 recs, 0.86 Mchars) σήμερον παρακατάσχω τὴν χρείαν, τέως τῆς δίκης περιγένωμαι· 
- `ia:FieldII1Co` (53 recs, 0.82 Mchars) ἄτε ἐν μείζοσι καὶ ἐπὶ μείζοσι πλάττοντες τοὺς ἀνθρώπους οὕτ
- `oga:tlg4013.tlg005` (1 recs, 0.79 Mchars) Simplicius — In Aristotelis Libros De Anima Commentaria [Sp.?] (fort. auctore Prisciano Lydo)
- `ia:commentariainar07alexgoog` (42 recs, 0.70 Mchars) ἀντεπακτὸς γένηται, οὔτε συκοφαντικὸς οὔτε ἐπακτὸς γίνεται· 
- `oga:tlg0057.tlg077` (1 recs, 0.66 Mchars) Galen — De compositione medicamentorum per genera

## fold 3 — 66967 records, 25639 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 42713 | 81.3 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `oga:tlg0060.tlg001` (3 recs, 2.60 Mchars) Diodorus Siculus — Historical Library
- `ia:operaineditaqupr00proc` (103 recs, 1.60 Mchars) προέδειξε, καὶ ἐκ τούτου αὖθις ταὐτὸν πεπονθότα εἴη· ἐστὶ δὲ
- `oga:tlg4029.tlg001` (1 recs, 1.44 Mchars) Procopius — De Bellis
- `oga:tlg0612.tlg001` (1 recs, 1.12 Mchars) Dio Chrysostom — Orationes
- `ia:bub_gb_djsOAAAAYAAJ` (61 recs, 1.00 Mchars) τεγκλήματος ἡ τοπικὴ καταδρομὴ, οὕτω καὶ ἀντιστάσεως ἡ τῶν σ
- `oga:tlg4013.tlg003` (1 recs, 0.99 Mchars) Simplicius — In Aristotelis Categorias Commentarium
- `oga:tlg2018.tlg005` (1 recs, 0.97 Mchars) Eusebius of Caesarea — Demonstratio Evangelica
- `oga:tlg0003.tlg001` (1 recs, 0.96 Mchars) Thucydides — History of the Peloponnesian War
- `greek_pd:u-vol.-2` (2125 recs, 0.88 Mchars) Τὸ δὲ τὸν θάνατον καὶ τὸν ἅδην εἰς τὴν λίμνην τοῦ πυρὸς βάλλ
- `ia:origenes-werke.-bd.-13-2015` (64 recs, 0.81 Mchars) τοῦτο ἐν τῷ Δευτερονομίῳ καὶ ἐν τῇ Ἐξόδῳ λέγεται. Οἶμαι γὰρ 
- `greek_pd:chrysostom_matthew_homilies_field_vol_1` (424 recs, 0.71 Mchars) πλέον. Κἂν γὰρ μὴ οὕτως, ἀλλ’ ἐκείνως ἀπολλύμεθα. Τί γὰρ δια
- `oga:tlg4102.tlg011` (1 recs, 0.71 Mchars) Catenae (Novum Testamentum) — Catena In Epistulam Ad Romanos (Typus Monacensis) (E Cod. Monac. gr. 412)

## fold 4 — 77803 records, 25637 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 53549 | 81.2 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `oga:tlg0081.tlg001` (1 recs, 1.86 Mchars) Dionysius of Halicarnassus — Antiquitates Romanae
- `greek_pd:tgl-prospectus-13` (17404 recs, 1.40 Mchars) Ἑκάτη παρὰ Χαλδαίοις. Λαϊῆς ἐν λαγόσιν κοινῆς Ἑκάτης ἀρετῆς 
- `oga:tlg2000.tlg001` (1 recs, 1.20 Mchars) Plotinus — Enneades
- `oga:tlg0732.tlg006` (1 recs, 1.07 Mchars) Alexander of Aphrodisias — In Aristotelis Topicorum Libros Octo Commentaria
- `oga:tlg2042.tlg001` (1 recs, 1.03 Mchars) Origen — Contra Celsum
- `ia:pg51chryso5` (57 recs, 1.00 Mchars) ηὑρέθημεν καὶ αὐτοὶ ἁμαρτωλοί, ἆρα Χριστὸς ἁμαρτίας διάκονος
- `ia:greek-spoken-lines-of-homeric-characters-3.0` (57 recs, 0.97 Mchars) Θρινακίην νήσωι, προφυγὼν ἰοειδέα πόντον, βοσκομένας δ’ εὕρη
- `ia:operaquaesupersu02demo` (56 recs, 0.91 Mchars) ὅτι, ἐκεῖ εἰ ἐτέθη, ὑπὲρ τοῦ Τιμοκράτους ἂν ἐτίθετο. ἰσχυρὰ 
- `ia:3_20260130_20260130_1457` (51 recs, 0.86 Mchars) σοι βασιλίδος θέαν περιεργάζεσθαι, φοβερὰ περὶ αὐτὴν ἡ τῶν δ
- `ia:novumtestamentum01hans` (53 recs, 0.84 Mchars) Ἰησοῦν περὶ τῶν μαθητῶν αὐτοῦ, καὶ περὶ τῆς διδαχῆς αὐτοῦ.Ἀπ
- `oga:tlg0551.tlg017` (1 recs, 0.75 Mchars) Appianus of Alexandria — Civil Wars
- `oga:tlg0012.tlg001` (1 recs, 0.68 Mchars) Homer — Iliad

## fold 5 — 64579 records, 25658 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 40325 | 80.5 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `greek_pd:GRCCRAM_DBS_HS` (5305 recs, 2.25 Mchars) τέρου μόνον θανάτου τῆς αἰωνίου κολάσεως. τὸ δὲ χορτασθῆναι 
- `oga:tlg0099.tlg001` (1 recs, 1.78 Mchars) Strabo — Geography
- `oga:tlg2018.tlg001` (1 recs, 1.43 Mchars) Eusebius of Caesarea — Praeparatio Evangelica
- `ia:b29320203` (80 recs, 1.27 Mchars) ΘΕΟΦ. Πολλάκις ἔμφραξις γίνεται περὶ τὰς ἀποφύσεις τῶν νεύρω
- `ia:Hilgard1901Scholia` (65 recs, 1.02 Mchars) σκοντα, ὅταν ἀναγινώσκῃ περί τινων ἐν ἀνάγκαις περιπεσόντων,
- `ia:librorumveterist00laga` (52 recs, 0.94 Mchars) πατέρων αὐτοῦ, καὶ ἔθαψαν αὐτὸν ἐν παραδείσῳ οἴκου αὐτοῦ, κα
- `oga:tlg2045.tlg001` (1 recs, 0.92 Mchars) Nonnus of Panopolis — Dionysiaca
- `ia:bub_gb_OE4BAAAAMAAJ` (54 recs, 0.83 Mchars) Ἔνθ’ ἐφάνη μέγα σῆμα, δράκων ἐπὶ νῶτα δαφοινός. Ἀημηγορία ἐσ
- `ia:elementa00eucl` (48 recs, 0.76 Mchars) ὄρου, ὁ τῶν δύο πρὸς τὰ τρία. ὁμοίως δὴ καὶ ἐπὶ πλειόνων, κα
- `ia:pg47chryso1` (41 recs, 0.70 Mchars) παιδίσκαι συνοικῶσιν αὐτῷ· τὴν δὲ ἄλλην ἀσχημοσύνην ἐξετάσωμ
- `greek_pd:alexandriaphrod00arisgoog` (343 recs, 0.66 Mchars) ταῦτά τε δὴ συμβαίνει πάντα καὶ τὸ ἐναντίον στοιχεῖον ἕως το
- `ia:adr3850.0004.001.umich.edu` (58 recs, 0.65 Mchars) κἀσπόδει: Ἔτυπτε. συνέτριβε συνέτριβεν, ἑκονιόρτου, ἑσπόδου 

## fold 6 — 62025 records, 25729 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 37771 | 80.5 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `oga:pta0003.pta020` (3 recs, 2.10 Mchars) πέφυκεν ὠφέλιμος ἡ βίβλος. ταύτην μετελθὼν καλὸν εὑρήσεις πλ
- `oga:tlg4015.tlg009` (1 recs, 1.71 Mchars) John Philoponus — In Aristotelis Physicorum Libros Commentaria
- `oga:tlg4031.tlg002` (1 recs, 1.45 Mchars) Eustratius — In Aristotelis Ethica Nicomachea I Commentaria
- `oga:tlg2018.tlg002` (2 recs, 1.31 Mchars) Eusebius of Caesarea — Historia Ecclesiastica
- `ia:pg53chryso7` (67 recs, 1.15 Mchars) ὁδοιπορίας κάματον αὐτῶν παραμυθήσασθαι. Εἶδες ὅση τοῦ δικαί
- `oga:tlg4015.tlg002` (1 recs, 0.95 Mchars) John Philoponus — In Aristotelis Analytica Priora Commentaria
- `oga:tlg2042.tlg005` (1 recs, 0.93 Mchars) Origen — Commentarii In Evangelium Joannis
- `greek_pd:aristotelisoper07arisuoft` (751 recs, 0.86 Mchars) τὸ πρέπον καλόν· πρέπει δὲ ταῦτα τούτῳ, πλοῦτος εὐγένεια δύν
- `ia:operaomniaquaeex10johnuoft` (97 recs, 0.85 Mchars) τὰς ἐκκλησιαστικὰς χρείας, καὶ εἰς τὸν λαὸν μηδὲν αὐθεντεῖν,
- `oga:tlg2048.tlg001` (1 recs, 0.75 Mchars) Sozomenus — Historia Ecclesiastica
- `ia:isaaksyr` (44 recs, 0.74 Mchars) ΛΟΓΟΣ ΟΒ Περιέχων ὑποθέσεις ὠφελίμους πεπληρωμένας τῆς σοφία
- `greek_pd:aristotelisorga02bekkgoog` (390 recs, 0.71 Mchars) Κατὰ μέρος μὲν οὖν περὶ ἑκάστης ἀρετῆς εἴρηται πρότερον ἐπεὶ

## fold 7 — 66485 records, 25692 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 42231 | 80.6 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `ia:schneider-procli-commentarius-in-platonis-timaeum-graece-gr-1847` (92 recs, 1.40 Mchars) ὅταν δόξα πρὸς δόξαν διαμάχηται, καὶ ὅταν ἢ κρείττων μὴ κατα
- `oga:tlg4015.tlg008` (1 recs, 1.37 Mchars) John Philoponus — In Aristotelis Libros De Anima Commentaria
- `ia:operacyrilli04cyriuoft` (70 recs, 1.11 Mchars) βαπτιστής· ἅγιος γὰρ ἦν καὶ ἐν ἐμβρύῳ καὶ μήτρᾳ. καὶ γοῦν ἐλ
- `ia:MN41481ucmf_2` (68 recs, 1.04 Mchars) Τί ἐστιν ἀνασταυροῦντας ἑαυτοῖς τὸν υἱὸν τοῦ θεοῦ καὶ παραδε
- `oga:pta0003.pta017` (1 recs, 1.00 Mchars) Eusebius of Caesarea — Commentarius in Isaiam
- `oga:tlg0057.tlg089` (1 recs, 0.99 Mchars) Galen — In Hippocratis Epidemiarum I
- `ia:chrysostom_matthew_homilies_field_vol_2` (62 recs, 0.95 Mchars) ἐκεῖνον περιῆς. Οὕτω καὶ ὁ ἀνὴρ ἐν ἀσφαλείᾳ, καὶ σὺ ἐν τιμῇ,
- `ia:bub_gb_YhYtAAAAMAAJ` (64 recs, 0.94 Mchars) ὡς μόνον καλουμένων αὐτῶν ἀφνειῶν, ἀλλ’ οὐκ ὄντων κατ’ ἀλήθε
- `ia:commentariiadho02eust` (51 recs, 0.86 Mchars) δηλοῖ, ὁ καὶ παραιτησάμενος ἐκεῖ διὰ τοῦτο φιλικὴν δόσιν ἵππ
- `oga:tlg0526.tlg004` (1 recs, 0.82 Mchars) Flavius Josephus — The Jewish War
- `ia:bub_gb_3bSjihp9L_0C` (77 recs, 0.71 Mchars) ρας ἐκείνας προσαγορεύουσι. Ταῦτά σοι εἰς προπομπὴν τοῦ αἰτε
- `ia:impjustinianippa01just` (46 recs, 0.70 Mchars) ται ταῖς αὐτοῦ κελεύσεσιν ὑπηρετούμενοι· λήψεται δὲ καὶ αὐτὸ

## fold 8 — 59710 records, 25661 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 35456 | 81.0 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `oga:tlg4090.tlg001` (1 recs, 1.91 Mchars) Cyril of Alexandria — In XII Prophetas
- `oga:tlg0732.tlg004` (1 recs, 1.90 Mchars) Alexander of Aphrodisias — In Aristotelis Metaphysica Commentaria
- `ia:pg48chryso2` (87 recs, 1.53 Mchars) Καὶ ὁ Ἄγαβος Τάδε λέγει τὸ Πνεῦμα τὸ ἅγιον· Τὸν ἄνδρα οὗ ἐστ
- `oga:tlg0057.tlg017` (1 recs, 1.20 Mchars) Galen — De usu partium corporis humani I-XI
- `ia:operaomniaquaeex07johnuoft` (130 recs, 1.16 Mchars) τιωτῶν πλῆθος; Μὴ γὰρ θύραν ἦν ἀνατρέψαι; μὴ γὰρ ἕνα λαθεῖν;
- `ia:operaomniaquaeex08johnuoft` (122 recs, 1.07 Mchars) μένη πλέον ἠσφαλίσατο· οὐ βούλεται γὰρ τὰς μὲν ἁμαρτίας τοῦ 
- `oga:tlg5026.tlg007` (1 recs, 0.97 Mchars) Scholia in Homerum — Scholia in Odysseam
- `ia:bub_gb_xU8BAAAAMAAJ` (59 recs, 0.96 Mchars) δικανικοῦ τέλος τὸ δίκαιον, τοῦ δὲ συμβουλευτικοῦ τὸ συμφέρο
- `oga:tlg0057.tlg076` (1 recs, 0.93 Mchars) Galen — De compositione medicamentorum secundum locos I-VI
- `oga:tlg0544.tlg002` (1 recs, 0.92 Mchars) Sextus Empiricus — Adversus Mathematicos
- `oga:tlg0057.tlg075` (1 recs, 0.84 Mchars) Galen — De simplicium medicamentorum temperamentis ac facultatibus I-VI
- `oga:tlg4102.tlg008` (1 recs, 0.74 Mchars) Catenae (Novum Testamentum) — Catena In Acta (Catena Andreae) (e cod. Oxon. coll. nov. 58)

## fold 9 — 60023 records, 25653 works/documents

| kind | records | Mchars |
|---|---|---|
| literary | 35769 | 81.3 |
| papyrus | 1949 | 0.8 |
| inscription | 22305 | 3.2 |

Largest test works:

- `oga:tlg0543.tlg001` (1 recs, 2.06 Mchars) Polybius — Histories
- `oga:tlg0526.tlg001` (1 recs, 2.03 Mchars) Flavius Josephus — Jewish Antiquities
- `ia:symeontheol` (93 recs, 1.59 Mchars) ποιοῦντα τινά, τῆς τῶν σῳζομένων σωτηρίας τυχεῖν.ΜΑ. Εὐχαρισ
- `oga:tlg0385.tlg001` (1 recs, 1.18 Mchars) Cassius Dio Cocceianus — Historia Romana
- `oga:tlg0016.tlg001` (1 recs, 1.17 Mchars) Herodotus — Histories
- `ia:operaomniaquaeex11johnuoft` (124 recs, 1.06 Mchars) Εἰς τὸν τετραήμερον Λάζαρον. γ.Ἔλυσε τὸν Λάζαρον τὸ φῶς· ἀλλ
- `ia:pg50chryso4` (62 recs, 1.05 Mchars) καθαρισθῶμεν, καὶ τὸν παράδεισον ἀπολάβωμεν. Ὢ μυστηρίου μεγ
- `oga:pta0005.pta002` (1 recs, 0.90 Mchars) Cyrill of Alexandria — Glaphyra in Pentateuchum
- `ia:aconcordanceto00mouluoft` (58 recs, 0.90 Mchars) περὶ τῆς ἐξόδου τ. υἱῶν Ἰσραὴλ ἐμνημόνευσεν ἠρνήσατο λέγεσθα
- `oga:tlg5023.tlg001` (1 recs, 0.89 Mchars) Scholia in Euripidem — Scholia in Euripidem (scholia vetera)
- `greek_pd:GRCSEPT_DBS_HS` (1511 recs, 0.75 Mchars) Κύριος παντοκράτωρ. Καὶ μακαριοῦσιν ὑμᾶς πάντα τὰ ἔθνη, διότ
- `greek_pd:bub_gb_BkS2KW7u76MC` (1248 recs, 0.70 Mchars) Παρμενίδης Παρμενίδης ὁ μέγας ὁ πατὴρ Παρμενίδης οἱ ἀμφὶ Παρ
