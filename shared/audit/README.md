# Pipeline Audit (code module)

Implementation lives in `shared/audit/` and `data/audit/`.

**Canonical documentation:** [`docs/audit/PIPELINE_AUDIT.md`](../../docs/audit/PIPELINE_AUDIT.md)

Quick entry points:

```python
from shared.audit import emit_audit_event
from data.audit.pipeline_events import query_audit_events, get_audit_events_for_structure
from data.audit.retention import run_audit_retention
```
