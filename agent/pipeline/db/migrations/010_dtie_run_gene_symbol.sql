-- Add gene_symbol to fact_dtie_run so protein-structure endpoint can
-- filter runs by gene rather than returning all structures.
ALTER TABLE fact_dtie_run ADD COLUMN IF NOT EXISTS gene_symbol TEXT;
CREATE INDEX IF NOT EXISTS idx_dtie_run_gene ON fact_dtie_run(gene_symbol);
