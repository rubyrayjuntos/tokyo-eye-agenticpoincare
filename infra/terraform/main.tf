# Tokyo Eye — Infrastructure
# Aurora PostgreSQL + S3 + supporting resources

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Aurora PostgreSQL (with pgvector)
# ---------------------------------------------------------------------------

resource "aws_rds_cluster" "tokyoeye" {
  cluster_identifier = "${var.project_name}-${var.environment}"
  engine             = "aurora-postgresql"
  engine_mode        = "provisioned"
  engine_version     = "16.1"
  database_name      = "tokyoeye"
  master_username    = var.db_master_username
  master_password    = var.db_master_password

  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 4.0
  }

  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.tokyoeye.name

  skip_final_snapshot = var.environment != "prod"
  deletion_protection = var.environment == "prod"

  tags = local.common_tags
}

resource "aws_rds_cluster_instance" "tokyoeye" {
  cluster_identifier = aws_rds_cluster.tokyoeye.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.tokyoeye.engine
  engine_version     = aws_rds_cluster.tokyoeye.engine_version

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# S3 — Object Storage for large artifacts
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project_name}-artifacts-${var.environment}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------------------------
# Networking (minimal)
# ---------------------------------------------------------------------------

resource "aws_security_group" "db" {
  name_prefix = "${var.project_name}-db-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.app_security_group_id]
  }

  tags = local.common_tags
}

resource "aws_db_subnet_group" "tokyoeye" {
  name       = "${var.project_name}-${var.environment}"
  subnet_ids = var.private_subnet_ids
  tags       = local.common_tags
}

# ---------------------------------------------------------------------------
# Locals
# ---------------------------------------------------------------------------

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
