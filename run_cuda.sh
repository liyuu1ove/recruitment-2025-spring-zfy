#!/bin/bash
#SBATCH -n 1
#SBATCH -o slurm-output/cuda/winograd-job-%j.out
#SBATCH -e slurm-error/cuda/winograd-job-%j.err
#SBATCH -c 64
#SBATCH --exclusive
#SBATCH --exclude hepnode0


# Note: How to run this script on slurm: `sbatch ./run.sh'.
# Note: see `man sbatch' for more options.

# Note: Manual control number of omp threads
# export OMP_NUN_THREADS=64

# Note: numactl - Control NUMA policy for processes or shared memory, see `man numactl'.`
# Note: perf-stat - Run a command and gather performance counter statistics, see `man perf stat'.

export NVTX_CUDA_PER_THREAD_TEMP_DIR=$SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR
ncu -o report ./winograd conf/vgg16.conf
# nsys profile ./winograd conf/vgg16.conf