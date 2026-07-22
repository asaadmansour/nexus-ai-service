# One image repository per service
resource "aws_ecr_repository" "repos" {
  for_each = toset(["nexus-ai-backend", "nexus-ai-frontend", "nexus-ai-service"])

  name         = each.value
  force_delete = true # lets `terraform destroy` remove them even with images

  image_scanning_configuration {
    scan_on_push = true
  }
}
