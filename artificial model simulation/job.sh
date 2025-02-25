#!/bin/bash

#PBS -l ncpus=27
#PBS -l mem=64GB
#PBS -l jobfs=1GB
#PBS -q normal
#PBS -P na4
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/na4+gdata/na4
#PBS -l wd

module load python3/3.11.7 openmpi/4.1.4
python3 test.py > $PBS_JOBID.log