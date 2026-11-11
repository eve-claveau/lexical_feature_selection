#!/bin/bash
#SBATCH --job-name=lexical_feature_selection
#SBATCH --account=def-yacineb
#SBATCH --time=12:00:00  
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/keywords_%j.out 

source /project/def-yacineb/openalex_snapshot/data_env/bin/activate

module load scipy-stack

echo "starting keyword extraction"

python3 multilingual_klr.py

echo "Done, tables in results/..."

