# -------------------------------
# Results bucket: run directories (results/<benchmark>/<system>/<label>_<ts>/...), per-pod chroma
# store snapshots (stores/<collection>/<version>.tar.zst) and job manifests (jobs/<job>/cells.json).
# Benchmark data itself stays in the data bucket (carnot-research), which is the source of truth.
# -------------------------------
resource "aws_s3_bucket" "results" {
  bucket = var.results_bucket

  tags = merge(local.common_tags, { Name = var.results_bucket })
}

resource "aws_s3_bucket_public_access_block" "results" {
  bucket = aws_s3_bucket.results.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Half-finished multipart uploads from a killed pod would otherwise sit around billing forever.
resource "aws_s3_bucket_lifecycle_configuration" "results" {
  bucket = aws_s3_bucket.results.id

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 2
    }
  }
}

# -------------------------------
# ECR repository for the experiment image (built by deploy_exp/docker/build_push.sh)
# -------------------------------
resource "aws_ecr_repository" "experiments" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE" # `latest` is re-pointed on every build; sha tags are still unique

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = local.common_tags
}

# The image is a few GB; keep the last 10 tagged builds and drop untagged layers after a day.
resource "aws_ecr_lifecycle_policy" "experiments" {
  repository = aws_ecr_repository.experiments.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "expire untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "keep the 10 most recent tagged images"
        selection = {
          tagStatus      = "tagged"
          tagPatternList = ["*"]
          countType      = "imageCountMoreThan"
          countNumber    = 10
        }
        action = { type = "expire" }
      },
    ]
  })
}
