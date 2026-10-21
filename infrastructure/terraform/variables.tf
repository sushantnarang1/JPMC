variable "name" {
  description = "Name prefix for the demonstration EKS cluster and VPC."
  type        = string
  default     = "modelops-hub"
}

variable "aws_region" {
  description = "AWS region for this example."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "Private VPC address range."
  type        = string
  default     = "10.40.0.0/16"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes control-plane version."
  type        = string
  default     = "1.30"
}

variable "cpu_instance_types" {
  description = "Managed CPU node group instance types."
  type        = list(string)
  default     = ["m6i.large"]
}

variable "enable_gpu_node_group" {
  description = "Whether to create an optional, initially scaled-to-zero GPU node group."
  type        = bool
  default     = false
}

variable "gpu_instance_types" {
  description = "GPU instance types supported in the selected AWS region."
  type        = list(string)
  default     = ["g5.xlarge"]
}
