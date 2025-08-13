#!/bin/bash	

#SBATCH --job-name=train_fungi
#SBATCH --output=train_fungi_result-%J.out
#SBATCH --cpus-per-task=1
#SBATCH --time=95:00:00
#SBATCH --mem=14gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=bmgi@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

conda init bash
source ~/.bashrc

source /opt/miniconda3/etc/profile.d/conda.sh

cd MultimodalDataChallenge2025/ 
conda activate mmss

echo -e "Working dir: $(pwd)\n"

python run.py train --checkpoint-folder=results --image-folder=/scratch/bmgi/FungiImages --metadata-folder=starting_metadata --model-config=configs/default_model_config.json --session=md_default --cuda


echo "Done: $(date +%F-%R:%S)"
