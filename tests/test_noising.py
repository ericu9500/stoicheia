"""Correctness properties for the elastic masked-diffusion noising."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.noising import NoiseConfig, noise_sequence, _pick_halfword_span


def _seq(n=64):
    g = torch.Generator().manual_seed(0)
    chars = torch.randint(0, 24, (n,), generator=g)
    # random word ends every ~5 chars, last is sentence-final
    boundary = torch.zeros(n, dtype=torch.uint8)
    i = 4
    while i < n - 1:
        boundary[i] = 1
        i += int(torch.randint(3, 7, (1,), generator=g).item())
    boundary[-1] = 2
    return chars, boundary


def test_labels_only_at_masked_positions_fixed_patterns():
    cfg = NoiseConfig(w_span=1, w_word=0, w_elastic=0, w_iid=0, w_halfword=0, w_substitute=0)
    g = torch.Generator().manual_seed(1)
    chars, boundary = _seq()
    for _ in range(50):
        out = noise_sequence(chars, boundary, cfg, g)
        assert not out["rebuilt"]
        masked = out["input_ids"] == cfg.mask_id
        supervised = out["labels"] != -100
        # every supervised position is masked, and its label is the original char
        assert torch.equal(masked, supervised)
        assert torch.equal(out["labels"][supervised], chars[supervised])
        # non-masked inputs are unchanged originals
        assert torch.equal(out["input_ids"][~masked], chars[~masked])


def test_iid_rate_matches_t():
    cfg = NoiseConfig(w_span=0, w_word=0, w_elastic=0, w_iid=1, w_halfword=0, w_substitute=0,
                      beta_a=1e6, beta_b=2e6)  # t ~ 1/3 tightly
    g = torch.Generator().manual_seed(2)
    fracs = []
    for _ in range(200):
        chars, boundary = _seq(256)
        out = noise_sequence(chars, boundary, cfg, g)
        fracs.append((out["input_ids"] == cfg.mask_id).float().mean().item())
    mean = sum(fracs) / len(fracs)
    assert 0.28 < mean < 0.39, mean


def test_word_masking_respects_boundaries():
    cfg = NoiseConfig(w_span=0, w_word=1, w_elastic=0, w_iid=0, w_halfword=0, w_substitute=0)
    g = torch.Generator().manual_seed(3)
    chars, boundary = _seq()
    ends = set((boundary >= 1).nonzero(as_tuple=True)[0].tolist())
    starts = {0} | {e + 1 for e in ends}
    for _ in range(50):
        out = noise_sequence(chars, boundary, cfg, g)
        masked = (out["input_ids"] == cfg.mask_id).tolist()
        # each maximal masked run must start at a word start and end at a word end
        i = 0
        while i < len(masked):
            if masked[i]:
                j = i
                while j + 1 < len(masked) and masked[j + 1]:
                    j += 1
                assert i in starts, f"masked run starts mid-word at {i}"
                assert j in ends, f"masked run ends mid-word at {j}"
                i = j + 1
            else:
                i += 1


def test_halfword_span_always_within_the_word_and_hits_all_anchors():
    """Unit-level: _pick_halfword_span must always return a sub-range of [s,e), and over many
    draws should hit all three anchors (begin/end/middle) — weighted toward the end (Greek is
    suffixal, so ending-restoration is the primary use case)."""
    cfg = NoiseConfig()
    g = torch.Generator().manual_seed(7)
    s, e = 100, 112   # a 12-char synthetic word
    touches_start = touches_end = interior_only = 0
    for _ in range(2000):
        a, b = _pick_halfword_span(s, e, cfg, g)
        assert s <= a < b <= e, f"halfword span [{a},{b}) escapes word range [{s},{e})"
        if a == s:
            touches_start += 1
        if b == e:
            touches_end += 1
        if a > s and b < e:
            interior_only += 1
    assert touches_start > 0 and touches_end > 0 and interior_only > 0
    assert touches_end > touches_start   # weighted toward the end (endings/inflection)


def test_halfword_pattern_only_masks_within_single_words():
    """Integration-level smoke test: halfword-only noising must not crash and every masked
    position must fall inside SOME word's char range (never the inter-word separator itself,
    which doesn't exist as a char anyway, but guards against off-by-one word-range bugs)."""
    cfg = NoiseConfig(w_span=0, w_word=0, w_elastic=0, w_iid=0, w_halfword=1, w_substitute=0)
    g = torch.Generator().manual_seed(9)
    chars, boundary = _seq(256)
    ends = sorted((boundary >= 1).nonzero(as_tuple=True)[0].tolist())
    starts = [0] + [e + 1 for e in ends[:-1]]
    in_word = torch.zeros(len(chars), dtype=torch.bool)
    for ws, we in zip(starts, [e + 1 for e in ends]):
        in_word[ws:we] = True
    for _ in range(100):
        out = noise_sequence(chars, boundary, cfg, g)
        masked = out["input_ids"] == cfg.mask_id
        assert bool((masked & ~in_word).any()) is False


def test_elastic_preserves_visible_chars_and_lengths():
    cfg = NoiseConfig(w_span=0, w_word=0, w_elastic=1, w_iid=0, w_halfword=0, w_substitute=0)
    g = torch.Generator().manual_seed(4)
    chars, boundary = _seq()
    for _ in range(80):
        out = noise_sequence(chars, boundary, cfg, g)
        assert out["rebuilt"]
        inp, lab = out["input_ids"], out["labels"]
        assert inp.numel() == lab.numel() == out["boundary"].numel()
        assert inp.numel() >= chars.numel()          # elastic only ever grows
        # visible (non-mask) positions carry original chars, in order
        visible = inp[inp != cfg.mask_id]
        # reconstruct target: concatenation of visible + gap targets must recover the original
        recon = []
        for tok, l in zip(inp.tolist(), lab.tolist()):
            if tok != cfg.mask_id:
                recon.append(tok)
            elif l != cfg.blank_id and l != -100:
                recon.append(l)
        assert recon == chars.tolist(), "elastic gap targets don't reconstruct the source"
        # every gap has at least the true chars then >=elastic_extra_min blanks
        blanks = (lab == cfg.blank_id).sum().item()
        assert blanks >= cfg.elastic_extra_min


def test_substitute_never_uses_mask_token_and_label_is_true_char():
    """DENOISING pattern: corrupted positions show a WRONG letter (never MASK), and the label
    is always the true original character, always different from what's shown."""
    cfg = NoiseConfig(w_span=0, w_word=0, w_elastic=0, w_iid=0, w_halfword=0, w_substitute=1)
    g = torch.Generator().manual_seed(11)
    chars, boundary = _seq(256)
    saw_any = False
    for _ in range(80):
        out = noise_sequence(chars, boundary, cfg, g)
        assert not out["rebuilt"]
        inp, lab = out["input_ids"], out["labels"]
        assert not (inp == cfg.mask_id).any(), "substitute must never emit the MASK token"
        corrupted = lab != -100
        if corrupted.any():
            saw_any = True
            # every corrupted position actually differs from the true char (genuinely wrong)
            assert bool((inp[corrupted] != chars[corrupted]).all())
            # label is always the true original character
            assert torch.equal(lab[corrupted], chars[corrupted])
        # untouched positions are unchanged originals
        assert torch.equal(inp[~corrupted], chars[~corrupted])
    assert saw_any


def test_channel_dropout_frequencies():
    """Per-position keep masks should average out to roughly the configured marginal rates
    (mixture of fully-known / fully-unknown / patchy-uniform-rate)."""
    cfg = NoiseConfig()
    g = torch.Generator().manual_seed(5)
    chars, boundary = _seq(256)
    kb_frac = kd_frac = kp_frac = 0.0
    N = 400
    for _ in range(N):
        out = noise_sequence(chars, boundary, cfg, g)
        kb_frac += out["keep_bnd_mask"].float().mean().item()
        kd_frac += out["keep_dia_mask"].float().mean().item()
        kp_frac += out["keep_punct_mask"].float().mean().item()
    # E[rate] = p_full*1 + p_none*0 + p_patchy*0.5
    exp_bnd = cfg.p_bnd_full + (1 - cfg.p_bnd_full - cfg.p_bnd_none) * 0.5
    exp_dia = cfg.p_dia_full + (1 - cfg.p_dia_full - cfg.p_dia_none) * 0.5
    exp_punct = cfg.p_punct_full + (1 - cfg.p_punct_full - cfg.p_punct_none) * 0.5
    assert abs(kb_frac / N - exp_bnd) < 0.07, kb_frac / N
    assert abs(kd_frac / N - exp_dia) < 0.07, kd_frac / N
    assert abs(kp_frac / N - exp_punct) < 0.07, kp_frac / N


def test_channel_masks_are_genuinely_patchy_sometimes():
    """At least some draws must have a channel PARTIALLY known (not all-or-nothing) — this is
    the actual point: some word-breaks/accents legible, others not, within one sequence."""
    cfg = NoiseConfig()
    g = torch.Generator().manual_seed(8)
    chars, boundary = _seq(256)
    patchy = 0
    for _ in range(300):
        out = noise_sequence(chars, boundary, cfg, g)
        m = out["keep_bnd_mask"]
        if 0 < m.float().mean().item() < 1:
            patchy += 1
    assert patchy > 0


def test_mixture_covers_all_patterns():
    cfg = NoiseConfig()
    g = torch.Generator().manual_seed(6)
    chars, boundary = _seq()
    rebuilt = fixed = 0
    for _ in range(200):
        out = noise_sequence(chars, boundary, cfg, g)
        rebuilt += out["rebuilt"]; fixed += not out["rebuilt"]
    assert rebuilt > 0 and fixed > 0


def _seq_with_lacuna(n=64, lac_start=20, lac_len=6):
    """A sequence with a real (whole-document) lacuna baked in: MASK_ID at [lac_start,
    lac_start+lac_len), matching insc/data/iphi.py's text_to_full_planes() convention."""
    chars, boundary = _seq(n)
    MASK_ID = 24
    chars = chars.clone()
    chars[lac_start:lac_start + lac_len] = MASK_ID
    boundary = boundary.clone()
    boundary[lac_start:lac_start + lac_len] = 3  # UNK_BND
    is_real_lacuna = torch.zeros(n, dtype=torch.bool)
    is_real_lacuna[lac_start:lac_start + lac_len] = True
    return chars, boundary, is_real_lacuna, lac_start, lac_len


def test_is_real_lacuna_none_is_fully_backward_compatible():
    """Passing is_real_lacuna=None (the default) must reproduce byte-identical output to the
    pre-existing call signature, for every existing caller.

    NOTE: _sample_t()'s Beta.sample() and the span pattern's Geometric.sample() don't actually
    consume the passed `g` --
    they draw from torch's GLOBAL RNG state instead. So reproducibility across two calls
    requires pinning torch.manual_seed() globally before each one, not just seeding two local
    generator objects identically."""
    cfg = NoiseConfig()
    chars, boundary = _seq()
    torch.manual_seed(42)
    g1 = torch.Generator().manual_seed(42)
    out1 = noise_sequence(chars, boundary, cfg, g1)
    torch.manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    out2 = noise_sequence(chars, boundary, cfg, g2, is_real_lacuna=None)
    assert torch.equal(out1["input_ids"], out2["input_ids"])
    assert torch.equal(out1["labels"], out2["labels"])


def test_real_lacuna_never_selected_as_synthetic_target_fixed_patterns():
    """span/word/halfword/iid/substitute must never choose a real-lacuna position as an
    ADDITIONAL synthetic-masking target -- its label must stay -100 and its input must stay
    exactly MASK_ID (never substituted to a wrong letter, never re-labeled)."""
    for w in ("w_span", "w_word", "w_halfword", "w_iid", "w_substitute"):
        cfg = NoiseConfig(**{"w_span": 0, "w_word": 0, "w_elastic": 0, "w_iid": 0,
                             "w_halfword": 0, "w_substitute": 0, w: 1})
        chars, boundary, is_real_lacuna, s, L = _seq_with_lacuna()
        g = torch.Generator().manual_seed(0)
        for trial in range(30):
            out = noise_sequence(chars, boundary, cfg, g, is_real_lacuna=is_real_lacuna)
            assert (out["labels"][s:s + L] == -100).all(), w
            assert (out["input_ids"][s:s + L] == 24).all(), w  # still exactly MASK_ID


def test_real_lacuna_excluded_from_elastic_rebuild_too():
    cfg = NoiseConfig(w_span=0, w_word=0, w_elastic=1, w_iid=0, w_halfword=0, w_substitute=0)
    chars, boundary, is_real_lacuna, s, L = _seq_with_lacuna()
    g = torch.Generator().manual_seed(0)
    for trial in range(30):
        out = noise_sequence(chars, boundary, cfg, g, is_real_lacuna=is_real_lacuna)
        # the lacuna's mask_id run must still be present somewhere, contiguous, with label -100
        ids = out["input_ids"]; lab = out["labels"]
        mask_positions = (ids == 24).nonzero(as_tuple=True)[0]
        assert mask_positions.numel() >= L
        # every genuinely-MASK_ID position with label -100 exists (the real lacuna survives
        # copy-through unlabeled even though the sequence may have grown/shrunk elsewhere)
        assert ((ids == 24) & (lab == -100)).sum() >= L
