import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_avg_max_SIC
import torch
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
    avg_max_SIC_list = []
    name_list = []
    
    for name in args.names:
    
        input_files = []
        for input_dir in args.input:
            input_files.extend(glob.glob(f"{input_dir}/plot_sig_inj_{name}/max_SIC_{name}.npz"))

        print(f"Files loaded for {name}.")

        sig_percent = np.load(input_files[0])["sig_percent"]

        max_SIC_list = [np.load(f)["max_SIC"] for f in input_files]
        
        avg_max_SIC = np.mean(np.stack(max_SIC_list, axis=0), axis=0)

        avg_max_SIC_list.append(avg_max_SIC)
        name_list.append(name)
    
    ntrains = int(len(args.input))
    plot_avg_max_SIC(sig_percent, avg_max_SIC_list, name_list, outdir=f"{args.outdir}", title=f"Average max SIC for {ntrains} trainings", tag=f"{ntrains}")
    
    
if __name__ == "__main__":
    main()