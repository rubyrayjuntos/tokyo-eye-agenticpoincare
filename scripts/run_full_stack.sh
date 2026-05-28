#!/bin/bash
# Tokyo Eyes — Full Stack Startup Script
# Run from the monorepo root: ./scripts/run_full_stack.sh
#
# This script:
# 1. Starts the database
# 2. Runs migrations
# 3. Ingests a structure (4OBE)
# 4. Starts the agent coordinator
# 5. Starts the visualizer frontend
#
# Prerequisites:
# - Docker running
# - Python 3.11+ with: psycopg[binary], pydantic, fastapi, uvicorn, python-jose, bcrypt, biotite
# - Node.js 18+ (for visualizer)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

echo "============================================"
echo "  Tokyo Eyes — Full Stack Startup"
echo "============================================"
echo ""
echo "Working directory: $PROJECT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Start database
# ---------------------------------------------------------------------------
echo "[1/5] Starting PostgreSQL (pgvector)..."
docker compose up -d db
echo "  Waiting for database to be ready..."
for i in $(seq 1 30); do
    if docker compose exec db pg_isready -U tokyoeye -d tokyoeye_dev > /dev/null 2>&1; then
        echo "  ✓ Database is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ✗ Database did not start in 30 seconds"
        exit 1
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# Step 2: Run migrations
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Running database migrations..."
python scripts/dev_setup.py migrate
echo "  ✓ Migrations complete"

# ---------------------------------------------------------------------------
# Step 3: Ingest structure
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Ingesting 4OBE from RCSB..."
python -c "
import asyncio
from data.db import get_connection, DBAdapter
from science.dtie.common.ingestion import StructureIngestor

async def main():
    async with get_connection() as conn:
        db = DBAdapter(conn)
        ingestor = StructureIngestor(db=db)
        result = await ingestor.ingest_pdb('4OBE', include_atoms=True)
        if result.warnings and 'Already ingested' in result.warnings:
            print('  ⊘ 4OBE already ingested')
        else:
            print(f'  ✓ Ingested: {result.structure_id} ({result.chains_created} chains, {result.residues_created} residues, {result.atoms_created} atoms)')

asyncio.run(main())
"

# ---------------------------------------------------------------------------
# Step 4: Start agent coordinator (background)
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Starting agent coordinator on port 8000..."

# Kill any existing agent process on port 8000
lsof -ti:8000 2>/dev/null | xargs -r kill 2>/dev/null || true
sleep 1

uvicorn agent.coordinator.app:app --host 0.0.0.0 --port 8000 --log-level info &
AGENT_PID=$!
echo "  ✓ Agent started (PID: $AGENT_PID)"

# Wait for agent to be ready
for i in $(seq 1 10); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ Agent responding at http://localhost:8000"
        break
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# Step 5: Start visualizer frontend (background)
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Starting visualizer frontend..."

if [ ! -d "visualizer/frontend/node_modules" ]; then
    echo "  Installing npm dependencies..."
    (cd visualizer/frontend && npm install --silent)
fi

(cd visualizer/frontend && npm run dev) &
FRONTEND_PID=$!
echo "  ✓ Visualizer starting (PID: $FRONTEND_PID)"
sleep 3

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Tokyo Eyes is running!"
echo "============================================"
echo ""
echo "  Agent API:    http://localhost:8000"
echo "  Agent Docs:   http://localhost:8000/docs"
echo "  Visualizer:   http://localhost:3000"
echo "  Database:     postgresql://tokyoeye:tokyoeye_dev_local@localhost:5432/tokyoeye_dev"
echo ""
echo "  Chat:         POST http://localhost:8000/api/chat"
echo "  Pipeline:     POST http://localhost:8000/api/tools/run-pipeline"
echo "  Poincaré:     GET  http://localhost:8000/api/poincare-data?structure_id=4obe"
echo ""
echo "  Press Ctrl+C to stop all services"
echo ""

# Trap Ctrl+C to clean up
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $AGENT_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    docker compose stop db
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for background processes
wait
