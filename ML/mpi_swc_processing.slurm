#!/bin/bash
#SBATCH --job-name=SWC_Processing
#SBATCH --output=SWC_Processing_%j.out
#SBATCH --error=SWC_Processing_%j.err
#SBATCH --partition=normal
#SBATCH --mem=100G                 # Total memory for the job
#SBATCH --time=7-00:00:00          # Time allocation (7 days)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xxxxxxx@gmu.edu
#SBATCH --ntasks=10   #  workers, 1 manager

set echo
umask 0027

# Print GPU status (if GPUs are requested)
#nvidia-smi

# Load required modules
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load openmpi/4.1.2-4a

# Activate the Conda environment
source ~/.bashrc
conda activate SWC_GCN_ENV

# Run the MPI program with processes
mpirun -n 10 python mpi_swc_processing.py
