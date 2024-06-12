#!/bin/bash
#SBATCH --account=nordtemhub
#SBATCH --partition=CPUQ
#SBATCH --time=00:30:00
#SBATCH --mem=5G
#SBATCH --reservation=nordtemhub
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="workshop_testing"

echo "we are running from this directory: ${SLURM_SUBMIT_DIR}"
echo "The name of the job is: $SLURM_JOB_NAME"

echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"

echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
source /cluster/projects/itea_lille-nv-fys-tem/miniforge3/bin/activate pyxem0.19.0
python test.py


scontrol show job ${SLURM_JOB_ID} -d
