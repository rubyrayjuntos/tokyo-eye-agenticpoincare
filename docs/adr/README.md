# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Tokyo Eye Agentic Poincaré platform.

## Purpose

ADRs capture important architectural decisions, including the context, the decision itself, and the consequences. They provide a historical record and help new contributors understand why the system is designed the way it is.

## Process

1. When a significant architectural decision needs to be made, create a new ADR using the template.
2. Number ADRs sequentially (ADR-001, ADR-002, etc.).
3. Mark the status as:
   - **Proposed** — Under discussion
   - **Accepted** — Decision made and implemented (or ready to be)
   - **Superseded** — Replaced by a later ADR
4. Update the status of any related ADRs when relevant.

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| ADR-001 | Residue as Primary Granular Anchor | Accepted | 2026-05-27 |
| ADR-002 | Parallel v3 and v4 Scientific Lineages | Accepted | 2026-05-27 |
| ADR-003 | Strict Separation of Training and Inference Outputs | Accepted | 2026-05-27 |

## Template

See `ADR_TEMPLATE.md` for the standard format.

---

**Note:** These ADRs are particularly focused on the data architecture, as data governance and the residue-centric extensible model are the highest priority for this platform.