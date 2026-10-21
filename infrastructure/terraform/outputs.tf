output "cluster_name" {
  description = "EKS cluster identifier."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Private EKS API endpoint."
  value       = module.eks.cluster_endpoint
}

output "vpc_id" {
  description = "VPC identifier."
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet identifiers for cluster workloads."
  value       = module.vpc.private_subnets
}
