import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div_data_reweight, plot_SIC_lists, plot_max_SIC, plot_rej_lists
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
    help="Input directory",
)
parser.add_argument(
    "-n",
    "--name",
    action="store",
    default="",
    help="Input directory",
)
parser.add_argument(
    "-k",
    "--kldiv",
    action="store_true",
    default=False,
    help="Plot kl div",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="Verbose enable DEBUG",
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

log_level = logging.DEBUG if args.verbose else logging.INFO
    
log = logging.getLogger("run")
log.setLevel(log_level)


def main():
    
    rundir = f"{args.input}/{args.name}"
    input_prefix = "idealAD_" if args.name=="idealAD" else ""

    all_files = glob.glob(f"{args.input}/{input_prefix}inputs_s*.npz")
    n_files = len(all_files)

    if n_files==0:
        sys.exit(f"No input files {args.input}/{input_prefix}inputs_s*.npz found! Exiting...")
    
    input_files = [f"{args.input}/{input_prefix}inputs_s{i}.npz" for i in range(n_files)]
    
    outdir = f"{args.input}/plot_sig_inj_{args.name}"
    os.makedirs(outdir, exist_ok=True)

    tpr_list = []
    fpr_list = []
    sig_percent_list = []
    
    for i in range(n_files):
        tpr_list.append(np.load(f"{rundir}/run{i}/signal_significance/tpr.npy"))
        fpr_list.append(np.load(f"{rundir}/run{i}/signal_significance/fpr.npy"))
        inputs = np.load(input_files[i])
        sig_percent_list.append(inputs["sig_percent"].item())
        inputs.close()
        
    
    plot_SIC_lists(tpr_list, fpr_list, sig_percent_list, name=f"{args.name}", outdir=outdir)
    
    plot_rej_lists(tpr_list, fpr_list, sig_percent_list, name=f"{args.name}", outdir=outdir)
    
    max_SIC_list = []

    if args.name=="idealAD":
        ideal_bkg_events_SR = np.load(input_files[0])["ideal_bkg_events_SR"]
        num_bkg_events = len(ideal_bkg_events_SR)
    else:
        MC_mask_SR = np.load(input_files[0])["MC_mask_SR"]
        num_bkg_events = np.sum(MC_mask_SR)

    for tpr, fpr in zip(tpr_list, fpr_list):
        
        sic = tpr[fpr > 0] / np.sqrt(fpr[fpr > 0])

        if num_bkg_events > 0:
            eps_bkg = 1.0/((0.4**2)*num_bkg_events)
            fpr_cutoff_indices = np.where(fpr[fpr > 0] > eps_bkg)
            max_SIC_list.append(np.nanmax(sic[fpr_cutoff_indices]))
        else:
            max_SIC_list.append(np.nanmax(sic))

    plot_max_SIC(sig_percent_list, max_SIC_list, label=f"{args.name}", outdir=outdir)


    # save SICs
    np.savez(f"{outdir}/max_SIC_{args.name}.npz", sig_percent=sig_percent_list, max_SIC=max_SIC_list)
    
if __name__ == "__main__":
    main()