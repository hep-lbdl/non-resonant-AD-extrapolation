import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import *
from semivisible_jet.utils import *
import os
import sys

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
    "-b2",
    "--ideal-bkg-dir",
    action="store",
    help="Input ideal bkground .txt files",
)
parser.add_argument(
    "-mc",
    "--mc-dir",
    action="store",
    help="Input MC bkground .txt files",
)
parser.add_argument(
    "--size",
    action="store",
    type=int,
    help="Input ideal bkground .txt files",
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

def check_file_log(bkg_path, ideal_bkg_path, mc_path):

    for file_path in [bkg_path, ideal_bkg_path, mc_path]:
        if not os.path.isfile(file_path):
            print(f"{file_path} does not exist!")


def main():

    # Create the output directory
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # define sample size as the number of files
    sample_size = args.size
    print(f"loadging {sample_size} samples...")

    # load signal first
    var_names = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    variables, sig = load_samples(args.sigsample)
    sig = get_quality_events(sig)
    sig_events = sort_event_arr(var_names, variables, sig)

    bkg_events_list = []
    ideal_bkg_events_list = []
    mc_events_list = []

    for i in range(sample_size):
        
        bkg_path = f"{args.bkg_dir}/qcd_{i}.txt"
        ideal_bkg_path = f"{args.ideal_bkg_dir}/qcd_{i}.txt"
        mc_path = f"{args.mc_dir}/qcd_{i}.txt"

        if os.path.isfile(bkg_path) and os.path.isfile(ideal_bkg_path) and os.path.isfile(mc_path):
            
            # Load input events ordered by varibles
            _, bkg_i = load_samples(bkg_path)
            _, ideal_bkg_i = load_samples(ideal_bkg_path)
            _, mc_i = load_samples(mc_path)
            
            bkg_i = get_quality_events(bkg_i)
            ideal_bkg_i = get_quality_events(ideal_bkg_i)
            mc_i = get_quality_events(mc_i)

            # Reshape bkg datasets
            bkg_datasets = reshape_bkg_events(bkg_i, ideal_bkg_i, mc_i)
            bkg_i = bkg_datasets[0]
            ideal_bkg_i = bkg_datasets[1]
            mc_i = bkg_datasets[2]

            bkg_events_list.append(sort_event_arr(var_names, variables, bkg_i))
            ideal_bkg_events_list.append(sort_event_arr(var_names, variables, ideal_bkg_i))
            mc_events_list.append(sort_event_arr(var_names, variables, mc_i))
        else:
            check_file_log(bkg_path, ideal_bkg_path, mc_path)

    if len(bkg_events_list)==0:
        sys.exit("No files loaded. Exit...")

    # concatenate all backgroud events
    bkg_events = np.concatenate(bkg_events_list)
    ideal_bkg_events = np.concatenate(ideal_bkg_events_list)
    mc_events = np.concatenate(mc_events_list)
    
    print(f"N total bkg_events: {bkg_events.shape[0]:.1e}")

    # SR masks
    bkg_mask_SR = apply_SR_cuts(bkg_events)
    ideal_bkg_mask_SR = apply_SR_cuts(ideal_bkg_events)
    MC_mask_SR = apply_SR_cuts(mc_events)

    # CR masks
    bkg_mask_CR = np.logical_not(bkg_mask_SR)
    MC_mask_CR = np.logical_not(MC_mask_SR)
    
    # SR events
    bkg_events_SR = bkg_events[bkg_mask_SR]
    ideal_bkg_events_SR = ideal_bkg_events[ideal_bkg_mask_SR]

    # initialize lists
    sig_percent_list = np.logspace(np.log10(0.001), np.log10(0.05), 8).round(5) - 0.001
    # sig_percent_list = [0]

    # Create signal injection dataset
    num = 0
    for s in sig_percent_list:
        
        # Subsample siganl set
        n_bkg_SR = bkg_events[bkg_mask_SR].shape[0]
        n_sig = int(s * n_bkg_SR)
        selected_sig_indices = np.random.choice(sig_events.shape[0], size=n_sig, replace=False)
        selected_sig = sig_events[selected_sig_indices, :] 

        # Create data arrays
        data_events  = np.concatenate([selected_sig, bkg_events])

        # SR and CR masks
        sig_mask_SR = apply_SR_cuts(selected_sig)
        data_mask_SR = apply_SR_cuts(data_events)
        
        sig_mask_CR = np.logical_not(sig_mask_SR)
        data_mask_CR = np.logical_not(data_mask_SR)

        # SR events
        sig_events_SR = selected_sig[sig_mask_SR]
        data_events_SR = data_events[data_mask_SR]
        rs = round(sig_events_SR.shape[0]/bkg_events_SR.shape[0], 5)

        # Print dataset information
        print(f"S/B={rs} in SR, N data SR: {data_events_SR.shape[0]:.1e}, N bkg SR: {bkg_events_SR.shape[0]:.1e}, N sig SR: {sig_events_SR.shape[0]}")
        
        # Plot varibles
        sig_list = sig_events_SR.T
        bkg_list = bkg_events_SR.T
        data_list = data_events_SR.T
        plot_dir = f"{outdir}/plots"
        os.makedirs(plot_dir, exist_ok=True)
        # Signal vs background
        plot_kwargs = {"name":f"sig_vs_bkg_SR_{rs}", "title":"Signal vs background in SR", "outdir":plot_dir}
        plot_all_variables(sig_list, bkg_list, var_names, **plot_kwargs)
        # data vs background SR
        plot_kwargs = {"labels":["data", "bkg"], "name":f"data_vs_bkg_SR_{rs}", "title":"Data vs background in SR", "outdir":plot_dir}
        plot_all_variables(data_list, bkg_list, var_names, **plot_kwargs)

        # Save ideal AD dataset
        np.savez(f"./{args.outdir}/idealAD_inputs_s{num}.npz", ideal_bkg_events_SR=ideal_bkg_events_SR, data_events_SR=data_events_SR, sig_percent=rs)  
        
        # Save extrapolation dataset
        sig_context = selected_sig[:, :2]
        sig_feature = selected_sig[:, 2:]
        bkg_context = bkg_events[:, :2]
        bkg_feature = bkg_events[:, 2:]
        MC_context = mc_events[:, :2]
        MC_feature = mc_events[:, 2:]
        data_context = data_events[:, :2]
        data_feature = data_events[:, 2:]
        np.savez(f"./{args.outdir}/inputs_s{num}.npz", data_feature=data_feature, data_context=data_context, MC_feature=MC_feature, MC_context=MC_context, bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, data_mask_CR=data_mask_CR, data_mask_SR=data_mask_SR, MC_mask_CR=MC_mask_CR, MC_mask_SR=MC_mask_SR, bkg_mask_CR=bkg_mask_CR, bkg_mask_SR=bkg_mask_SR, sig_mask_CR = sig_mask_CR, sig_mask_SR = sig_mask_SR, sig_percent=rs)
    
        num += 1
        
        
    print("Finished generating dataset.")

if __name__ == "__main__":
    main()