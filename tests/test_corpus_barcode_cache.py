"""Unit tests for dehydron barcode corpus cache-key suffix."""

from __future__ import annotations

from experiments.training.v6.corpus import _barcode_cache_suffix


def test_barcode_cache_suffix_explicit_false_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("USE_DEHYDRON_BARCODE", "1")
    monkeypatch.setenv("USE_BINNED_DEHYDRON", "1")
    assert _barcode_cache_suffix(use_dehydron_barcode=False) == ""


def test_barcode_cache_suffix_env_enabled_when_kwargs_none(monkeypatch) -> None:
    monkeypatch.setenv("USE_DEHYDRON_BARCODE", "1")
    monkeypatch.delenv("USE_BINNED_DEHYDRON", raising=False)
    assert _barcode_cache_suffix() == "|dbh_v1"


def test_barcode_cache_suffix_explicit_binned_true(monkeypatch) -> None:
    monkeypatch.setenv("USE_DEHYDRON_BARCODE", "0")
    monkeypatch.setenv("USE_BINNED_DEHYDRON", "0")
    assert (
        _barcode_cache_suffix(use_dehydron_barcode=True, use_binned_dehydron=True)
        == "|dbh_v1_binned"
    )
