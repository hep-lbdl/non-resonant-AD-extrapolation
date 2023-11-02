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
    help="Input bkground .txt file",
)
parser.add_argument(
    "-b2",
    "--bkgsample2",
    action="store",
    help="Input bkground .txt file",
)
parser.add_argument(
    "-mc",
    "--mcsample",
    action="store",
    help="Input MC bkground .txt file",
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
    help="output directory",
)
parser.add_argument(
    '-su', 
    "--supervised",
    action="store_true",
    default=False,
    help='Generate supervised datasets.'
)
parser.add_argument(
    '-id', 
    "--ideal",
    action="store_true",
    default=False,
    help='Generate supervised datasets.'
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
    _, MC = load_samples(args.mcsample)

    sig = get_quality_events(sig)
    bkg1 = get_quality_events(bkg1)
    bkg2 = get_quality_events(bkg2)
    MC = get_quality_events(MC)    
    
    # Reshape bkg datasets
    bkg_datasets = reshape_bkg_events(bkg1, bkg2, MC)
    bkg1 = bkg_datasets[0]
    bkg2 = bkg_datasets[1]
    MC = bkg_datasets[2]
    
    # Create context array
    var_names = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    sig_events = sort_event_arr(var_names, variables, sig)
    bkg_events = sort_event_arr(var_names, variables, bkg1)
    ideal_bkg_events = sort_event_arr(var_names, variables, bkg2)
    mc_events = sort_event_arr(var_names, variables, MC)

    # Define SR and CR masks
    HT_cut = 800    # In SR, HT > 800 GeV
    MET_cut = 75    # In SR, MET > 75 GeV
    
    bkg_mask_SR = (bkg_events[:, 0] > HT_cut) & (bkg_events[:, 1] > MET_cut)
    bkg_mask_CR = np.logical_not(bkg_mask_SR)

    ideal_bkg_mask_SR = (ideal_bkg_events[:, 0] > HT_cut) & (ideal_bkg_events[:, 1] > MET_cut)

    MC_mask_SR = (mc_events[:, 0] > HT_cut) & (mc_events[:, 1] > MET_cut)
    MC_mask_CR = np.logical_not(MC_mask_SR)
    
    # SR events
    bkg_events_SR = bkg_events[bkg_mask_SR]
    ideal_bkg_events_SR = ideal_bkg_events[ideal_bkg_mask_SR]

    # initialize lists
    if args.supervised:
        sig_percent_list = [1]*10
    else:
        # sig_percent_list = [0.1]
        sig_percent_list = np.logspace(np.log10(0.005),np.log10(0.1),10).round(5)
        # sig_percent_list = [0]
    
    # Create signal injection dataset
    num = 0
    for s in sig_percent_list:
        
        if args.supervised:
            selected_sig = sig_events
        else:
            # Subsample siganl set
            n_bkg_SR = bkg_events[bkg_mask_SR].shape[0]
            n_sig = int(s * n_bkg_SR)
            selected_sig_indices = np.random.choice(sig_events.shape[0], size=n_sig, replace=False)
            selected_sig = sig_events[selected_sig_indices, :] 


        # Create data arrays
        data_events  = np.concatenate([selected_sig, bkg_events])

        sig_mask_SR = (selected_sig[:, 0] > HT_cut) & (selected_sig[:, 1] > MET_cut)
        sig_mask_CR = np.logical_not(sig_mask_SR)

        data_mask_SR = (data_events[:, 0] > HT_cut) & (data_events[:, 1] > MET_cut)
        data_mask_CR = np.logical_not(data_mask_SR)
        

        # SR
        sig_events_SR = selected_sig[sig_mask_SR]
        data_events_SR = data_events[data_mask_SR]
        rs = round(sig_events_SR.shape[0]/bkg_events_SR.shape[0], 3)

        # Print dataset information
        print(f"Total dataset: N sig={len(selected_sig)}, N bkg={len(bkg_events)}, N idealAD bkg={len(ideal_bkg_events)}, N MC={len(mc_events)}")
        
        print(f"S/B={rs} in SR, N data SR: {data_events_SR.shape[0]}, N bkg SR: {bkg_events_SR.shape[0]}, N sig SR: {sig_events_SR.shape[0]}")
        
        # Plot varibles
        sig_list = sig_events_SR.T
        bkg_list = bkg_events_SR.T
        data_list = data_events_SR.T
        # Signal vs background
        plot_kwargs = {"name":f"sig_vs_bkg_SR_{rs}", "title":"Signal vs background in SR", "outdir":args.outdir}
        plot_all_variables(sig_list, bkg_list, var_names, **plot_kwargs)
        # data vs background SR
        plot_kwargs = {"labels":["data", "bkg"], "name":f"data_vs_bkg_SR_{rs}", "title":"Data vs background in SR", "outdir":args.outdir}
        plot_all_variables(data_list, bkg_list, var_names, **plot_kwargs)

        # Save dataset
        if args.supervised:
            np.savez(f"./{args.outdir}/supervised_inputs_{num}.npz", bkg_events_SR=bkg_events_SR, sig_events_SR=sig_events_SR, sig_percent=rs)
        elif args.ideal:
            np.savez(f"./{args.outdir}/idealAD_inputs_s{num}.npz", ideal_bkg_events_SR=ideal_bkg_events_SR, data_events_SR=data_events_SR, sig_percent=rs)
        else:
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