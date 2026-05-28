# Visualizer server — thin serving layer for the Poincaré disc viewer
"""Serves GNN embedding data and receives viewport directives.

This server:
1. Queries the governed data layer for embedding/uncertainty data
2. Formats it for the React frontend (JSON with disc coordinates)
3. Accepts viewport directives from the agent coordinator
4. Pushes updates to connected frontends via WebSocket
"""
