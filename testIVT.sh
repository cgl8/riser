#!/bin/bash
#PBS -P bp00
#PBS -q gpuvolta
#PBS -l walltime=05:00:00
#PBS -l mem=50GB
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l wd
#PBS -M u7644495@anu.edu.au
#PBS -m abe
#PBS -l storage=scratch/bp00+gdata/bp00

module load python3/3.9.2
source /g/data/bp00/jay/riser/.venvnew/bin/activate
SCRIPT='/g/data/bp00/jay/riser/riser/test4s.py'

python3 $SCRIPT
