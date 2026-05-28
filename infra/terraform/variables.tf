variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "tokyoeye"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "db_master_username" {
  description = "Aurora master username"
  type        = string
  sensitive   = true
}

variable "db_master_password" {
  description = "Aurora master password"
  type        = string
  sensitive   = true
}

variable "vpc_id" {
  description = "VPC ID for database placement"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for DB subnet group"
  type        = list(string)
}

variable "app_security_group_id" {
  description = "Security group ID of the application layer (allowed to connect to DB)"
  type        = string
}
