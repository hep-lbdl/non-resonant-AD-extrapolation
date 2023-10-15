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
        
        if ("phi" in x) or ("Phi" in x) or ("eta" in x):
            continue
            
        ind_x = ind(variables, x)
        sig_x = sig[ind_x]
        bkg_x = bkg[ind_x]
        title = f"{names[x]} distribution, min$p_{{\\rm T}} = {args.pTmin}$ GeV"
        xlabel = f"{names[x]} {units[x]}"
        
        if "tau" in x:
            bins = np.linspace(0, 1, 20)
        elif x=="met":
            bins = np.linspace(0, 600, 26)
        elif x=="ht":
            bins = np.linspace(0, 4000, 26)
        else:
            bins = None
        
        print(f"Plotting {x}")
        print(f"Num. of signal events: {len(sig_x)}")
        print(f"Num. of background events: {len(bkg_x)}")
        print("\n")
        
        # plot_quantity_list([sig_x, bkg_x], labels_list, title, xlabel, bins, x, args.outdir)
        plot_quantity_list_ratio([sig_x, bkg_x], labels_list, title, xlabel, bins, x, args.outdir)
    
        if x=="ht":
            ht_bkg = bkg_x

        if x=="met":
            met_bkg = bkg_x
    
    plot_correlation_hist(ht_bkg, met_bkg, "HT (GeV)", "MET (GeV)", "MET vs HT in QCD dijet", figname="_met_ht", outdir=args.outdir)
    
    
if __name__ == "__main__":
    main()
