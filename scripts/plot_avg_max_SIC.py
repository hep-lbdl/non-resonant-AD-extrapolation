import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import plot_avg_max_SIC
from scipy import stats
import os
import sys
import logging
import glob


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    action="store",
    nargs='+',
    help="Input directory",
)
parser.add_argument(
    "-n",
    "--names",
    action="store",
    nargs='+',
    default="",
    help="Input directory",
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
    help="output directory",
)
args = parser.parse_args()


def main():
    
    os.makedirs(args.outdir, exist_ok=True)
    
    sig_percent = []
    med_max_SIC_list = []
    lb_list = []
    ub_list = []
    name_list = []
    ntrains = 0
    
    for name in args.names:
    
        input_files = []
        for input_dir in args.input:
            input_files.extend(glob.glob(f"{input_dir}/plot_sig_inj_{name}/max_SIC_{name}.npz"))

        print(f"Total of {len(input_files)} files loaded for {name}.")
        ntrains = int(len(input_files))

        sig_percent = np.load(input_files[0])["sig_percent"]

        max_SIC_list = [np.load(f)["max_SIC"] for f in input_files]
        max_SIC_arr = np.stack(max_SIC_list, axis=0)

        # Lower bound of the uncertainty band: -1 sigma
        lb = np.percentile(max_SIC_arr, 15.87, axis=0)

        # Upper bound of the uncertainty band: + 1 sigma
        ub = np.percentile(max_SIC_arr, 84.13, axis=0)
        
        # Median max SIC
        med_max_SIC = np.percentile(max_SIC_arr, 50, axis=0)
        
        med_max_SIC_list.append(med_max_SIC)
        lb_list.append(lb)
        ub_list.append(ub)
        name_list.append(name)
    
    plot_avg_max_SIC(sig_percent, med_max_SIC_list, lb_list, ub_list, name_list, outdir=f"{args.outdir}", title=f"Max SIC for {ntrains} trainings", tag=f"{ntrains}")
    
    
if __name__ == "__main__":
    main()