"""Rebuild evaluator-ready run dirs from the released downstream checkpoints (offline).

  $STOICHEIA_DATA/hub_pt/doc_clean.pt                      backbone (architecture source for the loaders)
  released/parser_released/{best.pt,vocab.json}     -> python -m parser.joint_evaluate --run ...
  released/macronizer_released/best.pt              -> python -m meter.predict --model ... --norma
  released/meter_released/best.pt                   -> python -m meter.predict --model ... --norma
The released state dicts use the trainers' own key names, so they load strictly."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import torch
from safetensors.torch import load_file

R = Path(os.environ["STOICHEIA_DATA"])
HUB, OUT = R / "hub", R / "released"
bb = R / "hub_pt/doc_clean.pt"
if not bb.exists():
    subprocess.run([sys.executable, str(Path(__file__).with_name("hf_local_to_pt.py")), str(HUB / "Stoicheia-doc_clean"), str(bb)], check=True)

meta = json.load(open(HUB / "Stoicheia-tagger-parser/training_metadata.json"))
d = OUT / "parser_released"; d.mkdir(parents=True, exist_ok=True)
cfg = dict(meta["cfg"], ckpt=str(bb), kfold_dir=str(R / "treebanks/oga_repo/kfold"), out_dir=str(d))
torch.save(dict(model=load_file(str(HUB / "Stoicheia-tagger-parser/model.safetensors")), cfg=cfg,
                tcfg=meta["tcfg"], pcfg=meta["pcfg"], pretrain_cfg=meta.get("pretrain_cfg"), epoch=meta["epoch"],
                dev=meta["dev"], W=384, T=2048,
                deprel_vocab=json.load(open(HUB / "Stoicheia-tagger-parser/deprel_vocab.json"))), d / "best.pt")
shutil.copyfile(HUB / "Stoicheia-tagger-parser/vocab.json", d / "vocab.json")
print("parser ok", meta["dev"])

for rep, name in (("Stoicheia-macronizer", "macronizer_released"), ("Stoicheia-meter", "meter_released")):
    meta = json.load(open(HUB / rep / "training_metadata.json"))
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    msd = load_file(str(HUB / rep / "model.safetensors"))
    # the release omits MeterModel's class-weight buffers; they are fixed by mcfg
    msd.setdefault("mac_w", torch.tensor(meta["mcfg"]["mac_class_w"], dtype=torch.float32))
    msd.setdefault("scan_w", torch.tensor(meta["mcfg"]["scan_class_w"], dtype=torch.float32))
    torch.save(dict(model=msd, cfg=dict(meta["cfg"], ckpt=str(bb)),
                    mcfg=meta["mcfg"], pretrain_cfg=meta.get("pretrain_cfg"), epoch=meta.get("epoch"),
                    dev=meta.get("dev"), T=meta.get("T", 2048)), d / "best.pt")
    print(name, "ok", meta.get("dev"))
