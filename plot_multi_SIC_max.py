import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div_data_reweight, plot_SIC_lists, plot_multi_max_SIC
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
    "--name",
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
    
    if len(args.input) != len(args.name):
    
        raise RuntimeError("Lengths of input and name lists do not match.")

    else:
        
        input_files = []
        for input_dir in args.input:
            input_files.extend(glob.glob(f"{input_dir}/max_SIC_*.npz"))

        name_list = args.name
        
        print(f"Files loaded! Input contains {name_list}.")
    
    
    sig_percent = np.logspace(np.log10(0.001),np.log10(0.05),10).round(4)
    
    max_SIC_list = [np.load(f)["max_SIC"] for f in input_files]
    
    plot_multi_max_SIC(sig_percent, max_SIC_list, name_list, outdir=f"{args.outdir}")

    
if __name__ == "__main__":
    main()