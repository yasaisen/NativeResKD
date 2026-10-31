#!/bin/bash
#SBATCH --job-name=NRKD_v4_01
#SBATCH --partition=8gpus
#SBATCH --account=MST114560
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yasaisenkiri@gmail.com

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Allocated node(s): ${SLURM_NODELIST}"
echo "Num GPUs: ${SLURM_GPUS_PER_NODE}"

source /home/misaka13/anaconda3/etc/profile.d/conda.sh
conda activate MVLM_NRKD

NRKD_PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${NRKD_PROJECT_DIR}"
mkdir -p logs

python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port="$((29500 + RANDOM % 500))" \
  R27_MVLM_trainNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config_hpc.json

# keepTrain 範例：
# python -m torch.distributed.run --nproc_per_node=4 --master_port="$((29500 + RANDOM % 500))" \
#   R27_MVLM_trainNRKD_v0.0.py --latest-checkpoint-path /work/misaka13/R27_MVLM_v4.01/checkpoints/YYMMDDHHMMSS --use-this-dir
