# Tokyo Eyes v3 Package Contract

This directory freezes the Phase 0 package contract for Tokyo Eyes v3.

## Canonical policy

- `structure.cif` is canonical and immutable.
- Tokyo Eyes outputs are sidecars.
- `augmented_structure.cif` is optional derived output.

## Required files

- `structure.cif`
- `tokyoeyes_annotations.parquet` (or `tokyoeyes_annotations.json` during early prototyping)
- `tokyoeyes_model_output.json`
- `provenance.json`
- `manifest.json`

## Optional files

- `graph.pt`
- `augmented_structure.cif`

## Schemas and templates

- JSON Schemas: `schemas/`
- Starter templates: `templates/`

## Validation

Use:

```bash
python scripts/validate_tokyoeyes_package.py --package-dir /path/to/package
```

This validates required files, core fields, pointers, and `sha256` checksums declared in `manifest.json`.
