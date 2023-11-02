import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div_data_reweight, plot_SIC_lists, plot_max_SIC
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
        
    
    if args.kldiv:
        for i in range(n_files):

            # load input files
            inputs = np.load(input_files[i])
            # data and MC
            data_feature = inputs["data_feature"]
            data_mask_SR = inputs["data_mask_SR"]
            data_mask_CR = inputs["data_mask_CR"]
            MC_feature = inputs["MC_feature"]
            MC_mask_SR = inputs["MC_mask_SR"]
            inputs.close()

            data_SR = data_feature[data_mask_SR]
            data_CR = data_feature[data_mask_CR]
            MC_SR = MC_feature[MC_mask_SR]

            samples_path = f"{args.input}/run{i}/samples_data_feat_SR.npz"
            pred_bkg_SR = np.load(samples_path)["samples"]
            #pred_bkg_SR_from_truth = np.load(samples_path)["samples_from_truth"]
            w_MC = np.load(samples_path)["weights"]

            title_tag=f"$x=N(0.5(m_1 + m_2), 1)$ with S/B={sig_percent_list[i]}"

            # Plot true data in SR, predicted background in SR, reweighted background in SR, and the training data in CR.
            plot_kl_div_data_reweight(data_CR, data_SR, pred_bkg_SR, w_MC, name=f"data_reweight_s{i}", title=title_tag, ymin=-8, ymax=16, outdir=outdir)
        
    
    plot_SIC_lists(tpr_list, fpr_list, sig_percent_list, name=f"{args.name}", outdir=outdir)
    
    max_SIC_list = [np.max(tpr[fpr > 0] / np.sqrt(fpr[fpr > 0])) for tpr, fpr in zip(tpr_list, fpr_list)]
    
    plot_max_SIC(sig_percent_list, max_SIC_list, label=f"{args.name}", outdir=outdir)
    
    # save SICs
    np.savez(f"{outdir}/max_SIC_{args.name}.npz", sig_percent=sig_percent_list, max_SIC=max_SIC_list)
    
if __name__ == "__main__":
    main()