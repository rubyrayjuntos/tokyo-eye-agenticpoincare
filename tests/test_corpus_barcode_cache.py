"""Unit tests for dehydron barcode corpus cache-key suffix."""

from __future__ import annotations

import os

from experiments.training.v6.corpus import _barcode_cache_suffix


def test_barcode_cache_suffix_explicit_false_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("USE_DEHYDRON_BARCODE", "1")
    monkeypatch.setenv("USE_BINNED_DEHYDRON", "1")
    assert _barcode_cache_suffix(use_dehydron_barcode=False) == ""


def test_barcode_cache_suffix_env_enabled_when_kwargs_none(monkeypatch) -> None:
    monkeypatch.setenv("USE_DEHYDRON_BARCODE", "1")
    monkeypatch.delenv("USE_BINNED_DEHYDRON", raising=False)
    assert _barcode_cache_suffix() == "|dbh_v1|dbh_sidecars_none"


def test_barcode_cache_suffix_explicit_binned_true(monkeypatch) -> None:
    monkeypatch.setenv("USE_DEHYDRON_BARCODE", "0")
    monkeypatch.setenv("USE_BINNED_DEHYDRON", "0")
    assert (
        _barcode_cache_suffix(use_dehydron_barcode=True, use_binned_dehydron=True)
        == "|dbh_v1_binned|dbh_sidecars_none"
    )


def test_barcode_cache_suffix_changes_when_sidecar_added_or_replaced(tmp_path) -> None:
    first = _barcode_cache_suffix(
        use_dehydron_barcode=True,
        use_binned_dehydron=False,
        dehydron_barcode_dir=tmp_path,
    )
    assert first.endswith("|dbh_sidecars_none")

    sidecar = tmp_path / "4OBE_A_dehydron_barcode_v1.pt"
    sidecar.write_bytes(b"first")
    os.utime(sidecar, ns=(1_000_000_000, 1_000_000_000))
    with_sidecar = _barcode_cache_suffix(
        use_dehydron_barcode=True,
        use_binned_dehydron=False,
        dehydron_barcode_dir=tmp_path,
    )

    sidecar.write_bytes(b"replacement-payload")
    os.utime(sidecar, ns=(2_000_000_000, 2_000_000_000))
    replaced = _barcode_cache_suffix(
        use_dehydron_barcode=True,
        use_binned_dehydron=False,
        dehydron_barcode_dir=tmp_path,
    )

    assert with_sidecar != first
    assert replaced != with_sidecar
