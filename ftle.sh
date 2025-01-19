#!/bin/bash

########SBATCH -A ctb-steinman
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=80
#SBATCH --time=23:59:00
######SBATCH --time=1:00:00
######SBATCH -p debug
#SBATCH --job-name ftle_case_A
#SBATCH --output=ftle_A_%j.txt

export MPLCONFIGDIR=/scratch/s/steinman/ahaleyyy/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ahaleyyy/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ahaleyyy/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true

module load CCEnv StdEnv/2020 gcc/9.3.0 python/3.9.6 #petsc/3.20.0

source $HOME/.virtualenvs/pv_updated/bin/activate 

mpiexec -np 80 python ftle.py ../mesh_rez/cases/case_A/case_028_low/results/art_*
