#!/bin/bash

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J reweight_10
#SBATCH --mail-user=rmastand@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 8:00:00
#SBATCH -A m3246

for g in 1 2 3 4 5 6 7 8 9 10
do
    for s in 0.008 0.016
    do
        python run_reweight.py -i /global/cfs/cdirs/m3246/rmastand/bkg_extrap/mod_mc/ -s $s -g $g
    done
done
