# Migrated from: SRC_AGENT/main.py on 2026-05-27
"""Agent coordinator — FastAPI service that orchestrates scientific workflows.

The coordinator:
1. Accepts user requests (chat, API calls)
2. Plans multi-step scientific workflows
3. Calls DTIE tools to execute science
4. Emits ViewportDirectives to the visualizer
5. Manages session state and provenance context
"""
