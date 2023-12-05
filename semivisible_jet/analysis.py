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
    "-mc",
    "--mcsample",
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
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # Get the events ordered by varibles
    variables, sig = load_samples(args.sigsample)
    variables, bkg = load_samples(args.bkgsample)
    variables, mc = load_samples(args.mcsample)
    
    labels_list = [r"$Z' \to jj$, $r_{\rm inv} = 1/3$", "QCD dijet data"]
    labels_list_mc = [r"$Z' \to jj$, $r_{\rm inv} = 1/3$", "QCD dijet data", "QCD dijet MC"]
    
    names = name_map()
    units = unit_map()
    
    for x in variables:
        
        if ("phi" in x) or ("Phi" in x) or ("eta" in x):
            continue
            
        ind_x = ind(variables, x)
        sig_x = sig[:, ind_x]
        bkg_x = bkg[:, ind_x]
        mc_x = mc[:, ind_x]
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
        
        plot_quantity_list_ratio([sig_x, bkg_x], labels_list, title, xlabel, bins, x, args.outdir)
        plot_quantity_list([sig_x, bkg_x, mc_x], labels_list_mc, title, xlabel, bins, x, args.outdir)
    
        if x=="ht":
            ht_bkg = bkg_x

        if x=="met":
            met_bkg = bkg_x
    
    plot_correlation_hist(ht_bkg, met_bkg, "HT (GeV)", "MET (GeV)", 4000, 600, "MET vs HT in QCD dijet", figname="_met_ht", outdir=args.outdir)
    
    # Define SR and CR masks
    HT_cut = 800    # In SR, HT > 800 GeV
    MET_cut = 75    # In SR, MET > 75 GeV
    
    # Create context array
    context_names = ["ht", "met"]
    bkg_context = sort_event_arr(context_names, variables, bkg)
    
    bkg_mask_SR = (bkg_context[:, 0] > HT_cut) & (bkg_context[:, 1] > MET_cut)
    bkg_mask_CR = np.logical_not((bkg_context[:, 0] > HT_cut) & (bkg_context[:, 1] > MET_cut))
    
    print(bkg_context.shape)
    print(bkg_mask_SR.shape)
    print(bkg_mask_CR.shape)
    print(f"num event SR {np.sum(bkg_mask_SR)}")
    print(bkg_context[bkg_mask_SR].shape)
    
    plot_correlation_hist(ht_bkg[bkg_mask_CR], met_bkg[bkg_mask_CR], "HT (GeV)", "MET (GeV)", 4000, 600, "MET vs HT in QCD dijet CR", figname="_met_ht_CR", outdir=args.outdir)
    
    
if __name__ == "__main__":
    main()
