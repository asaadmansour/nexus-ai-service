# Minimal, low-cost alternative to the EKS stack in ../terraform:
# one EC2 in the default VPC running docker compose. No NAT, no load balancer, no control-plane fee.
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
  region = var.region
}

variable "region" {
  default = "us-east-1"
}

variable "key_name" {
  description = "Existing EC2 key pair used to SSH into the host"
  type        = string
}

variable "instance_type" {
  default = "t3.small"
}

# Default VPC keeps this cheap and simple (no NAT gateway needed)
data "aws_vpc" "default" {
  default = true
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "app" {
  name        = "nexus-ai-app"
  description = "Web + SSH for the single-host deployment"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP (also used for the TLS challenge)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # tighten to your own IP for real use
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Lets the host pull images from ECR without storing AWS keys on it
resource "aws_iam_role" "app" {
  name = "nexus-ai-app-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "app_ecr" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "app" {
  name = "nexus-ai-app-profile"
  role = aws_iam_role.app.name
}

# Static IP so DuckDNS keeps working across reboots
resource "aws_eip" "app" {
  instance = aws_instance.app.id
  domain   = "vpc"
}

resource "aws_instance" "app" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.app.name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 30
  }

  # Software is installed by the Ansible playbook, not here: unattended boot-time
  # apt runs hold the dpkg lock and make an immediately-following playbook fail.

  tags = { Name = "nexus-ai-app" }
}

# One image repository per service. GitHub Actions pushes here; the host pulls.
resource "aws_ecr_repository" "repos" {
  for_each = toset(["nexus-ai-backend", "nexus-ai-frontend", "nexus-ai-service"])

  name         = each.value
  force_delete = true # lets `terraform destroy` remove them even with images

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the 10 newest images so storage costs stay flat
resource "aws_ecr_lifecycle_policy" "expire_old" {
  for_each   = aws_ecr_repository.repos
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

output "public_ip" {
  description = "Point your DuckDNS domain at this IP"
  value       = aws_eip.app.public_ip
}

output "ecr_registry" {
  description = "Set this as the ECR_REGISTRY GitHub secret and in the host .env"
  value       = split("/", values(aws_ecr_repository.repos)[0].repository_url)[0]
}

output "ssh" {
  value = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_eip.app.public_ip}"
}
