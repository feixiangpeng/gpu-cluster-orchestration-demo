#!/bin/bash
#SBATCH --job-name=priority_demo
#SBATCH --output=priority_%j.out
#SBATCH --error=priority_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --account=research_project

#SBATCH --qos=high
#SBATCH --priority=100
#SBATCH --nice=0

echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running with priority settings: QOS=high, Priority=100"

echo "Starting high priority job simulation..."
sleep 30
echo "High priority job component completed"

mkdir -p results
echo "Job $SLURM_JOB_ID completed with priority 100" > results/priority_test.txt

echo "Job finished at: $(date)"