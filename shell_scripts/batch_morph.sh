#!/bin/bash

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J morph_10
#SBATCH --mail-user=rmastand@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 8:00:00
#SBATCH -A m3246

for s in 0.0026 0.0053 0.008 0.0106 0.0133 0.016
do
   python run_morph.py -i /global/cfs/cdirs/m3246/rmastand/bkg_extrap/mod_mc/ -s $s -g 10 -lb True -bm /global/cfs/cdirs/m3246/rmastand/bkg_extrap/mod_mc/models/seed1/morph_base_best_s0.pt
done

