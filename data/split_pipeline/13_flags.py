#!/usr/bin/env python3
"""Stage 13: build/refresh the Greek-origin flag list for Corpus Corporum.

A bronze passage is flagged when its source Latin work is (or likely is) a
translation/close paraphrase of a GREEK work, detected via:
  - corpus membership (Graeca miscellanea, Ptolemaeus Latinus,
    Versiones latinae, Libri sacri / Biblia)
  - Greek-writing author (name list + Greek-provenance epithets)
  - the SAME author patterns appearing in the work title / text_name
    (catches "Iliados liber XIV Latine redditus" by 'Anonymus', Poliziano's
    Latin Iliad, commentaries on Aristotle, etc.)
  - explicit translation markers in the title (latine redditus/versus,
    e/ex Graeco, interprete, versio)

Deliberately over-inclusive: flagging a Latin-original work costs a little
train data; missing a translation breaks the no-paraphrase claim.

Output: work/bronze_greek_origin_flags.json {cc_idno: bool}
"""
import json
import os
import re

import orjson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = os.path.expandvars("$STOICHEIA_ROOT/catalog.jsonl")

BAD_CORPORA = {"Graeca miscellanea", "Ptolemaeus Latinus", "Versiones latinae",
               "Libri sacri / Biblia"}

GREEK_AUTHORS = """
Plutarch Aristotel Plato Homer Hesiod Herodot Thucydid Xenophon Demosthen
Isocrat Aeschin Lysias Pindar Sophocl Euripid Aeschyl Aristophan Menand
Theocrit Callimach Apolloni Galen Hippocrat Dioscorid Ptolemae Euclid
Archimed Diophant Pappus Proclus Plotin Porphyri Iamblich Epictet
Theophrast Strabo Pausani Diodor Dionysius Halicarnass Polyb Appian
Arrian Cassius Dio Herodian Josephus Philo Lucian Longin Athenae
Origenes Clemens Alexandrin Eusebius Athanasi Basilius Gregorius Nazianzen
Gregorius Nyssen Chrysostom Cyrillus Alexandrin Cyrillus Hierosolymit
Epiphani Theodoret Johannes Damascen Maximus Confessor Ignatius Antiochen
Polycarp Irenae Justinus Martyr Hermas Barnabas Didym Anastasius Sinait
Sophroni Andreas Caesariens Oecumeni Theophylact Photius Suidas
Johannes Climac Ephraem Syrus Pseudo-Dionys Dionysius Areopagit
Nemesius Synesius Themisti Simplici Alexander Aphrodisiens Ammonius
Olympiodor Philopon Sextus Empiric Diogenes Laert Marcus Aurelius
Aesop Babrius Oppian Quintus Smyrnae Nonnus Musaeus Colluth Tryphiodor
Apollodor Antonin Liberal Partheni Chariton Achilles Tati Heliodor
Longus Xenophon Ephesi Aristid Aelian Artemidor Cebes Hierocl
Aratus Nicander Manetho Vettius Valens Hephaestio
Basilius Seleuciens Asterius Amasen Palladius Socrates Scholastic
Sozomen Evagri Zosim Procop Agathias Menander Protector Theophan
Constantinus Porphyrogen Anna Comnen Nicetas Choniat Michael Psell
Barlaam Bessarion Gemistus Pletho Chalcocondyl
Hero Alexandrin Aristarch Autolyc Theodosius Tripolit Geminus
Cleomedes Nicomach Theon Alexandrin Theon Smyrnae Hypsicl
Antiochus Monach Johannes Moschus Dorotheus Gazae Isaac Syrus
Macarius Aegypti Marcus Eremita Nilus Ancyran Diadochus Photicens
Hesychius Symeon Metaphrast Symeon Novus Theolog Germanus Constantinopolit
Tarasius Nicephorus Constantinopolit Theodorus Studit Leo Sapiens
Leontius Byzantin Johannes Malal Georgius Pisid Cosmas Indicopleust
Romanus Melod Andreas Cretens Johannes Euchait Eustathius Thessalonicens
Michael Glycas Johannes Cinnam Georgius Acropolit Georgius Pachymer
Theodorus Metochit Nicephorus Gregor Johannes Cantacuzen Demetrius Cydon
Manuel Palaeolog Marcus Eugenic Georgius Trapezunt Theodorus Gaza
Johannes Argyropul Demetrius Chalcondyl Marcus Musurus Janus Lascaris
Aeneas Gazae Zacharias Mitylen Johannes Philopon Stephanus Byzantin
Agapetus Diacon Johannes Carpathi Thalassius Theognost Philotheus Sinait
Elias Ecdic Petrus Damascen Theodorus Edessen Theodorus Abucara
Michael Syncell Ephraem Antiochen
Ilia Iliad Odyss Batrachomyom Argonautic Anacreont
""".split()

AUTHOR_RE = re.compile("|".join(re.escape(w) for w in GREEK_AUTHORS), re.I)
EPITHET_RE = re.compile(
    r"(Alexandrin|Constantinopolit|Nazianzen|Nyssen|Chrysostom|Damascen|"
    r"Areopagit|Byzantin|Thessalonicens|Hierosolymit|Antiochen|Caesariens|"
    r"Cappadoc|Atheniens|Ephesi|Smyrnae|Sinait|Studit|Metaphrast|Graec)", re.I)
TRANSL_RE = re.compile(
    r"(e Graeco|ex Graeco|Graece|interprete|translat\w* .{0,20}Graec|versio|"
    r"latine (reddit|vers|expr)|graecolatin)", re.I)


def main():
    flags = {}
    n_flag = 0
    w_flag = w_tot = 0
    with open(CATALOG, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            a = r.get("author") or ""
            title = "%s || %s" % (r.get("work") or "", r.get("text_name") or "")
            w = int(r.get("words") or 0)
            bad = (r["corpus"] in BAD_CORPORA
                   or AUTHOR_RE.search(a) or EPITHET_RE.search(a)
                   or AUTHOR_RE.search(title) or TRANSL_RE.search(title))
            flags[r["cc_idno"]] = bool(bad)
            n_flag += bool(bad)
            w_flag += w * bool(bad)
            w_tot += w
    with open(os.path.join(ROOT, "work", "bronze_greek_origin_flags.json"),
              "w") as f:
        json.dump(flags, f)
    print("flagged %d/%d works, %.1fM/%.1fM words (%.1f%%)" %
          (n_flag, len(flags), w_flag / 1e6, w_tot / 1e6,
           100.0 * w_flag / w_tot))


if __name__ == "__main__":
    main()
