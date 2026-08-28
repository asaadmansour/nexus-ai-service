variable "region" {
  default = "us-east-1"
}

variable "cluster_name" {
  default = "nexus-ai"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair (used to SSH into the Jenkins host)"
  type        = string
}
