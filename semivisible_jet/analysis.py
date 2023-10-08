import numpy as np
import matplotlib.pyplot as plt
from utils  import *
import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--sigsample",
    action="store",
    help="Path to the first signal sample .txt file.",
)
parser.add_argument(
    "-b",
    "--bkgsample",
    action="store",
    help="Path to the background sample .txt file.",
)
parser.add_argument(
    "-pt",
    "--pTmin",
    action="store",
    default="800",
    help="The minimum pT required.",
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
    outdir = args.outdir
    
    sig_output = np.loadtxt(args.sigsample, dtype=str)
    bkg_output = np.loadtxt(args.bkgsample, dtype=str)

    # Get the names of all varibles
    variables = sig_output[0]
    
    # Get the events ordered by varibles
    sig = np.asfarray(sig_output[1:]).T
    bkg = np.asfarray(bkg_output[1:]).T
    
    labels_list = [r"$Z' \to jj$, $r_{\rm inv} = 1/3$", "QCD dijet"]
    
    names = name_map()
    units = unit_map()
    
    for x in variables:
        
        if ("phi_" in x) or (x == "pT_j2") or ("eta" in x):
            continue
            
        ind_x = ind(variables, x)
        sig_x = sig[ind_x]
        bkg_x = bkg[ind_x]
        title = f"{names[x]} distribution, min$p_{{\\rm T}} = {args.pTmin}$ GeV"
        xlabel = f"{names[x]} {units[x]}"
        bins = np.linspace(0, 1, 20) if "tau" in x else None
        
        plot_quantity_list([sig_x, bkg_x], labels_list, title, xlabel, bins, x, args.outdir)
    
    
    
    
if __name__ == "__main__":
    main()
