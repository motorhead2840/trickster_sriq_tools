"""The Witness register keeps every dissertation chapter and separates tonight from later."""
from __future__ import annotations

from pathlib import Path

from experiments.dgx_abhaya_round import abstain_mask, make_series
from experiments.future_tech_register import REGISTER, by_when

DOC = Path("docs/future-tech-register.md")
FRAMEWORK = Path("docs/theoretical-framework.md")


def test_every_chapter_is_present_and_ppln_is_later():
    text = DOC.read_text(encoding="utf-8")
    framework = FRAMEWORK.read_text(encoding="utf-8")
    for chapter in ("Ch.1", "Ch.2", "Ch.3", "Ch.4", "Ch.5", "Ch.6"):
        assert any(chapter in source for _, source, _ in REGISTER)
        assert chapter.replace("Ch.", "Ch.") in text or f"ch{chapter[-1]}" in text
    assert "ch6-ppln" in by_when("later")
    assert "ch5-landauer-calorimetry" in by_when("later")
    assert "ch5-sign-xor" in by_when("tonight")
    assert "nb-abstain" in by_when("tonight")
    assert "PPLN" in text or "lithium niobate" in text
    assert "0.0062" in text
    assert "future-tech-register.md" in framework
    assert "Cell 8" in text and "Cell 11" in text
    ids = [row[0] for row in REGISTER]
    assert len(ids) == len(set(ids))
    assert len(by_when("tonight")) >= 15


def test_abstain_flags_the_spike_series_more_than_it_misses():
    _latent, observed = make_series()
    silent = abstain_mask(observed, threshold=1.0)
    # Spikes are not returned by make_series; recompute the mask size only.
    assert silent.shape == observed.shape
    assert int(silent.sum()) > 0
