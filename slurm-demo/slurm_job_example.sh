#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --account=research_project

module purge
module load cuda/11.7.0
module load python/3.9.0
module load pytorch/2.0.0

export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

source ~/envs/myenv/bin/activate

echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"

nvidia-smi

srun python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(hostname -s) \
    --master_port=12345 \
    train.py \
    --epochs 50 \
    --batch-size 128 \
    --model resnet50 \
    --data-path /scratch/datasets/imagenet \
    --output-dir /scratch/results/experiment1

echo "Job finished at: $(date)"