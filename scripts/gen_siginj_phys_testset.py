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
    "--bkgsample1",
    action="store",
    help="Input background .txt file",
)
parser.add_argument(
    "-b2",
    "--bkgsample2",
    action="store",
    help="Input background .txt file",
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
    

def reshape_bkg_events(bkg1, bkg2, MC):
    
    # Number of bkg events
    n_bkg = np.min([bkg1.shape[0], bkg2.shape[0], MC.shape[0]])

    datasets = [bkg1, bkg2, MC]

    for i in range(3):
        # Reshape bkg2 to match bkg1
        selected_indices = np.random.choice(datasets[i].shape[0], size=n_bkg, replace=False)
        datasets[i] = datasets[i][selected_indices, :] 

    return datasets


def main():

    # Create the output directory
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # Load input events ordered by varibles
    variables, sig = load_samples(args.sigsample)
    _, bkg1 = load_samples(args.bkgsample1)
    _, bkg2 = load_samples(args.bkgsample2)

    sig = get_quality_events(sig)
    bkg1 = get_quality_events(bkg1)
    bkg2 = get_quality_events(bkg2)
    
    # Create context array
    var_names = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    sig_events = sort_event_arr(var_names, variables, sig)
    bkg1_events = sort_event_arr(var_names, variables, bkg1)
    bkg2_events = sort_event_arr(var_names, variables, bkg2)
    
    # Add bkg datasets
    bkg_events = np.concatenate([bkg1_events, bkg2_events], axis=0)

    # Define SR and CR masks
    HT_cut = 800    # In SR, HT > 800 GeV
    MET_cut = 75    # In SR, MET > 75 GeV
    
    bkg_mask_SR = (bkg_events[:, 0] > HT_cut) & (bkg_events[:, 1] > MET_cut)
  
    # Select small number of signal for testing
    n_sig = bkg_events[bkg_mask_SR].shape[0]
    selected_sig_indices = np.random.choice(sig_events.shape[0], size=n_sig, replace=False)
    selected_sig = sig_events[selected_sig_indices, :]
    
    sig_mask_SR = (selected_sig[:, 0] > HT_cut) & (selected_sig[:, 1] > MET_cut)
    
    # SR events
    bkg_events_SR = bkg_events[bkg_mask_SR]
    sig_events_SR = selected_sig[sig_mask_SR]

    # Print dataset information
    print(f"Test dataset in SR: N sig={len(sig_events_SR)}, N bkg={len(bkg_events_SR)}")

    # Plot varibles
    sig_list = sig_events_SR.T
    bkg_list = bkg_events_SR.T
    plot_kwargs = {"name":f"sig_vs_bkg_testset", "title":f"N sig={len(sig_events_SR)}, N bkg={len(bkg_events_SR)}", "outdir":args.outdir}
    plot_all_variables(sig_list, bkg_list, var_names, **plot_kwargs)

    # Save dataset
    np.savez(f"./{args.outdir}/test_inputs.npz", bkg_events_SR=bkg_events_SR, sig_events_SR=sig_events_SR)
    
        
    print("Finished generating dataset.")

if __name__ == "__main__":
    main()