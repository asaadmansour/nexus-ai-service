output "jenkins_public_ip" {
  value = aws_instance.jenkins.public_ip
}

output "ecr_urls" {
  value = { for name, repo in aws_ecr_repository.repos : name => repo.repository_url }
}

output "configure_kubectl" {
  description = "Run this after apply to point kubectl at the cluster"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${var.cluster_name}"
}
