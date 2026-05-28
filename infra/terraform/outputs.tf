output "aurora_endpoint" {
  description = "Aurora cluster endpoint"
  value       = aws_rds_cluster.tokyoeye.endpoint
}

output "aurora_reader_endpoint" {
  description = "Aurora reader endpoint"
  value       = aws_rds_cluster.tokyoeye.reader_endpoint
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for large artifacts"
  value       = aws_s3_bucket.artifacts.id
}

output "database_url" {
  description = "Full database connection URL (sensitive)"
  value       = "postgresql://${var.db_master_username}:${var.db_master_password}@${aws_rds_cluster.tokyoeye.endpoint}:5432/tokyoeye"
  sensitive   = true
}
