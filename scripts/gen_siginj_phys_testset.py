import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import *
from semivisible_jet.utils import *
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--sigsample",
    action="store",
    help="Input signal .txt file",
)
parser.add_argument(
    "-b1",
    "--bkg-dir",
    action="store",
    help="Input bkground .txt files",
)
parser.add_argument(
    "--size",
    action="store",
    type=int,
    help="Input ideal bkground .txt files",
)
parser.add_argument(
    "-su",
    "--supervised",
    action="store_true",
    default=False,
    help="Generate supervised dataset",
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
    help="output directory",
)
args = parser.parse_args()


def get_quality_events(arr):

    if np.isnan(arr).any():
        return arr[~np.isnan(arr).any(axis=1)]
    
    else:
        return arr


def main():

    # Create the output directory
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # define sample size as the number of files
    sample_size = args.size
    
    # load signal first
    var_names = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    variables, sig = load_samples(args.sigsample)
    sig = get_quality_events(sig)
    sig_events = sort_event_arr(var_names, variables, sig)

    bkg_events_list = []

    for i in range(sample_size):
        
        # Load input events ordered by varibles
        file_path = f"{args.bkg_dir}/qcd_{i}.txt"
        if os.path.isfile(file_path):
            _, bkg_i = load_samples(file_path)
            bkg_i = get_quality_events(bkg_i)
            bkg_events_list.append(sort_event_arr(var_names, variables, bkg_i))

    # concatenate all backgroud events
    bkg_events = np.concatenate(bkg_events_list)

    # SR bkg
    bkg_mask_SR = apply_SR_cuts(bkg_events)
    bkg_events_SR = bkg_events[bkg_mask_SR]
    
    # Select small number of signal for testing
    if sig_events.shape[0] > bkg_events_SR.shape[0]:
        n_sig = bkg_events_SR.shape[0]
        selected_sig_indices = np.random.choice(sig_events.shape[0], size=n_sig, replace=False)
        selected_sig = sig_events[selected_sig_indices, :]
    else:
        selected_sig = sig_events
    
    # SR sig
    sig_mask_SR = apply_SR_cuts(selected_sig)
    sig_events_SR = selected_sig[sig_mask_SR]

    type_prefix = "supervised" if args.supervised else "test"

    # Print dataset information
    print(f"{type_prefix} dataset in SR: N sig={len(sig_events_SR)}, N bkg={len(bkg_events_SR)}")

    # Plot varibles
    sig_list = sig_events_SR.T
    bkg_list = bkg_events_SR.T
    plot_kwargs = {"name":f"sig_vs_bkg_{type_prefix}", "title":f"N sig={len(sig_events_SR)}, N bkg={len(bkg_events_SR)}", "outdir":args.outdir}
    plot_all_variables(sig_list, bkg_list, var_names, **plot_kwargs)

    # Save dataset
    np.savez(f"./{args.outdir}/{type_prefix}_inputs.npz", bkg_events_SR=bkg_events_SR, sig_events_SR=sig_events_SR)
    
        
    print(f"Finished generating {type_prefix} dataset.")

if __name__ == "__main__":
    main()