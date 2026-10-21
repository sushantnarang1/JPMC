terraform {
  required_version = ">= 1.5.0"
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

data "aws_availability_zones" "available" {
  state = "available"
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.8.1"

  name = "${var.name}-vpc"
  cidr = var.vpc_cidr
  azs  = slice(data.aws_availability_zones.available.names, 0, 3)

  private_subnets = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i)]
  public_subnets  = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i + 8)]

  enable_nat_gateway     = true
  single_nat_gateway     = true
  one_nat_gateway_per_az = false
  enable_dns_hostnames   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${var.name}" = "shared"
  }
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "20.8.5"

  cluster_name    = var.name
  cluster_version = var.kubernetes_version
  cluster_endpoint_public_access = false
  cluster_endpoint_private_access = true
  enable_cluster_creator_admin_permissions = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  cluster_encryption_config = {
    resources = ["secrets"]
  }
  cluster_security_group_additional_rules = {
    egress_all = {
      description      = "Allow cluster egress"
      protocol         = "-1"
      from_port        = 0
      to_port          = 0
      type             = "egress"
      cidr_blocks      = ["0.0.0.0/0"]
    }
  }

  eks_managed_node_groups = merge(
    {
      cpu = {
        name           = "${var.name}-cpu"
        instance_types  = var.cpu_instance_types
        min_size        = 1
        max_size        = 5
        desired_size    = 2
        subnet_ids      = module.vpc.private_subnets
        capacity_type   = "ON_DEMAND"
      }
    },
    var.enable_gpu_node_group ? {
      gpu = {
        name           = "${var.name}-gpu"
        instance_types  = var.gpu_instance_types
        min_size        = 0
        max_size        = 3
        desired_size    = 0
        subnet_ids      = module.vpc.private_subnets
        capacity_type   = "ON_DEMAND"
        labels = {
          workload = "gpu-inference"
        }
        taints = {
          dedicated = {
            key    = "nvidia.com/gpu"
            value  = "true"
            effect = "NO_SCHEDULE"
          }
        }
      }
    } : {}
  )
}
