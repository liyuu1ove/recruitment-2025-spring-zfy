#!/bin/bash
#SBATCH -n 1
#SBATCH -o slurm-output/small/winograd-job-%j.out
#SBATCH -e slurm-error/small/winograd-job-%j.err
#SBATCH -c 64
#SBATCH --exclusive
#SBATCH --exclude hepnode0

# Note: How to run this script on slurm: `sbatch ./run.sh'.
# Note: see `man sbatch' for more options.

# Note: Manual control number of omp threads
# export OMP_NUN_THREADS=64

# Note: numactl - Control NUMA policy for processes or shared memory, see `man numactl'.`
# Note: perf-stat - Run a command and gather performance counter statistics, see `man perf stat'.

numactl --cpunodebind=0-3 --membind=0-3 perf stat -ddd ./winograd conf/small.conf 