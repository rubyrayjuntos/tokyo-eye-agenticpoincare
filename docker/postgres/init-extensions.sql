-- Tokyo Eye: Extensions initialized at DB creation time
-- These run once when the container first creates the database.

CREATE EXTENSION IF NOT EXISTS vector;          -- pgvector: embeddings, similarity search
CREATE EXTENSION IF NOT EXISTS plpython3u;      -- PL/Python: hyperbolic geometry UDFs (numpy)
CREATE EXTENSION IF NOT EXISTS pg_trgm;         -- Trigram: fuzzy text matching for agent recall
CREATE EXTENSION IF NOT EXISTS pg_cron;         -- Cron: scheduled materialized view refresh
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";     -- UUID generation for canonical keys
