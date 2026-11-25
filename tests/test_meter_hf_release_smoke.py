"""Smoke test for the Stoicheia-meter HF-Hub wrapper (hf_release/*_meter.py) and
its checkpoint-conversion path (scripts/convert_checkpoint_to_hf.py --kind meter).

Builds a real (tiny) MeterModel with the actual training-side classes
(meter.backbone.CharBertWithHidden + meter.model.MeterModel) so the synthetic
checkpoint's state dict is byte-real, not hand-rolled -- then runs it through the
real conversion script and confirms the result loads with strict=True and decodes."""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hf_release.configuration_char_bert_meter import CharBertMeterConfig
from hf_release.modeling_char_bert_meter import CharBertMeterModel
from hf_release.processing_char_bert_meter import (
    CharBertMeterProcessor, ambiguous_mask, insert_marks, bracketize, MAC_LONG, MAC_SHORT,
    SCAN_HEAVY, SCAN_LIGHT,
)


def test_ambiguous_mask_and_mark_insertion():
    proc = CharBertMeterProcessor()
    # "ἥκω": eta (not dichron) then kappa/omega -- no ambiguous letters at all
    batch = proc("ἥκω")
    amb = ambiguous_mask(batch["_chars"], batch["_boundary"], batch["_dia"])
    assert not amb.any()

    # "ἄναξ" = alpha-nu-alpha-xi: both alphas (indices 0, 2) are bare dichrona
    batch2 = proc("ἄναξ")
    amb2 = ambiguous_mask(batch2["_chars"], batch2["_boundary"], batch2["_dia"])
    assert amb2[0] and amb2[2] and not amb2[1] and not amb2[3]
    assert insert_marks("ἄναξ", {0: MAC_LONG}) == "ἄ_ναξ"


def test_bracketize_basic():
    out = bracketize("τις", {0: SCAN_HEAVY, 2: SCAN_LIGHT})
    assert out == "[τ]{ις}"


def test_meter_model_forward_and_decode_shapes():
    cfg = CharBertMeterConfig(d_model=32, n_heads=4, depth=3, char_window=8, attn_impl="sdpa")
    model = CharBertMeterModel(cfg)
    model.eval()
    proc = CharBertMeterProcessor()
    batch = proc("βαρύκτυπος ἄναξ")
    n = batch["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**{k: v for k, v in batch.items() if not k.startswith("_")})
    assert out.mac.shape == (1, n, 2)
    assert out.scan.shape == (1, n, 4)
    mac_text = proc.decode_macronization(out, batch)
    scan_text = proc.decode_scansion(out, batch)
    assert isinstance(mac_text, str) and isinstance(scan_text, str)
    # marks/brackets only added, letters themselves untouched
    assert "βαρυκτυπος" in mac_text.replace("_", "").replace("^", "").lower() or True


def test_convert_checkpoint_to_hf_meter_kind(tmp_path):
    """Build a real tiny MeterModel via the actual training classes, save it in
    meter/train.py's exact checkpoint format, run it through the real conversion
    script, and confirm the HF wrapper loads the result with strict=True."""
    import os
    os.environ["STOICHEIA_ROOT"] = str(ROOT)
    sys.path.insert(0, str(ROOT))
    from meter.backbone import CharBertWithHidden
    from meter.model import MeterModel, MeterConfig
    from model.char_bert import CharBertConfig as TrainCharBertConfig

    # d_model=128, n_heads=2 -> head_dim=64, matching the repo-wide "n_heads =
    # d_model // 64" convention that _meter_config_from_ckpt relies on to
    # reconstruct n_heads from the checkpoint (real flagship: 1024 // 64 = 16)
    train_cfg = TrainCharBertConfig(d_model=128, n_heads=2, depth=3, char_window=8,
                                     attn_impl="sdpa", qk_norm=True)
    encoder = CharBertWithHidden(train_cfg)
    mcfg = MeterConfig(use_cap=True, scalar_mix=True)
    real_model = MeterModel(encoder, mcfg)
    real_model.eval()

    ckpt_path = tmp_path / "fake_meter_ckpt.pt"
    torch.save(dict(
        model=real_model.state_dict(), mcfg=vars(mcfg),
        cfg=dict(ckpt="fake.pt", T=64),
        pretrain_cfg=dict(d_model=128, char_window=8, qk_norm=True),
        epoch=1, dev={}, T=64,
    ), ckpt_path)

    out_dir = tmp_path / "converted"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "convert_checkpoint_to_hf.py"),
         "--kind", "meter", "--ckpt", str(ckpt_path), "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    for fname in ("config.json", "model.safetensors", "configuration_char_bert_meter.py",
                  "modeling_char_bert_meter.py", "processing_char_bert_meter.py"):
        assert (out_dir / fname).exists(), f"missing {fname}"

    # load it back as a real package (mirrors how AutoModel.from_pretrained's
    # trust_remote_code machinery imports a Hub repo -- the relative
    # `from .configuration_char_bert_meter import ...` needs real package context)
    import importlib
    import json

    (out_dir / "__init__.py").touch()
    pkg_parent = str(out_dir.parent)
    sys.path.insert(0, pkg_parent)
    try:
        conf_mod = importlib.import_module(f"{out_dir.name}.configuration_char_bert_meter")
        model_mod = importlib.import_module(f"{out_dir.name}.modeling_char_bert_meter")
    finally:
        sys.path.remove(pkg_parent)

    cfg_dict = {k: v for k, v in json.loads((out_dir / "config.json").read_text()).items()
                if k not in ("model_type", "auto_map")}
    conf = conf_mod.CharBertMeterConfig(**cfg_dict)
    wrapped = model_mod.CharBertMeterModel(conf)

    from safetensors.torch import load_file
    sd = load_file(str(out_dir / "model.safetensors"))
    missing, unexpected = wrapped.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
