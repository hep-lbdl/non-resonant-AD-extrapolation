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
    context_names = ["ht", "met"]
    bkg_context = sort_event_arr(context_names, variables, bkg1)
    MC_context = sort_event_arr(context_names, variables, MC)
    ideal_bkg_context = sort_event_arr(context_names, variables, bkg2)
    
    # Create feature array
    feature_names = ["m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    bkg_feature = sort_event_arr(feature_names, variables, bkg1)
    MC_feature = sort_event_arr(feature_names, variables, MC)
    ideal_bkg_feature = sort_event_arr(feature_names, variables, bkg2)

    # Define SR and CR masks
    HT_cut = 800    # In SR, HT > 800 GeV
    MET_cut = 75    # In SR, MET > 75 GeV
    
    MC_mask_SR = (MC_context[:, 0] > HT_cut) & (MC_context[:, 1] > MET_cut)
    MC_mask_CR = np.logical_not(MC_mask_SR)
    
    bkg_mask_SR = (bkg_context[:, 0] > HT_cut) & (bkg_context[:, 1] > MET_cut)
    bkg_mask_CR = np.logical_not(bkg_mask_SR)

    ideal_bkg_mask_SR = (ideal_bkg_context[:, 0] > HT_cut) & (ideal_bkg_context[:, 1] > MET_cut)
    idea_bkg_mask_CR = np.logical_not(ideal_bkg_mask_SR)
    
    
    # initialize lists
    if args.supervised:
        sig_percent_list = [1]*10
    else:
        sig_percent_list = [0.1]
        #sig_percent_list = np.logspace(np.log10(0.005),np.log10(0.1),10).round(5)
        # sig_percent_list = [0]
    
    # Create signal injection dataset
    num = 0
    for s in sig_percent_list:
        
        if args.supervised:
            selected_sig = sig
        else:
            # Subsample siganl set
            n_sig = int(s * bkg1.shape[0])
            selected_sig_indices = np.random.choice(sig.shape[0], size=n_sig, replace=False)
            selected_sig = sig[selected_sig_indices, :] 
        
        sig_context = sort_event_arr(context_names, variables, selected_sig)
        sig_feature = sort_event_arr(feature_names, variables, selected_sig)
        
        sig_mask_SR = (sig_context[:, 0] > HT_cut) & (sig_context[:, 1] > MET_cut)
        sig_mask_CR = np.logical_not((sig_context[:, 0] > HT_cut) & (sig_context[:, 1] > MET_cut))

        # Create data arrays
        data_context = np.vstack([sig_context, bkg_context])
        data_feature = np.vstack([sig_feature, bkg_feature])

        data_mask_SR = (data_context[:, 0] > HT_cut) & (data_context[:, 1] > MET_cut)
        data_mask_CR = np.logical_not((data_context[:, 0] > HT_cut) & (data_context[:, 1] > MET_cut))
        
        # Print dataset information
        print(f"S/B={s}, N sig={len(selected_sig)}, N bkg={len(bkg1)}, N idealAD bkg={len(ideal_bkg_context)}, N MC={len(MC_context)}, N data SR: {data_context[data_mask_SR].shape[0]}")
        
        # Plot varibles
        sig_list = np.vstack([sig_context[sig_mask_SR].T, sig_feature[sig_mask_SR].T])
        bkg_list = np.vstack([bkg_context[bkg_mask_SR].T, bkg_feature[bkg_mask_SR].T])
        data_list = np.vstack([data_context[data_mask_SR].T, data_feature[data_mask_SR].T])
        # Signal vs background
        plot_kwargs = {"name":f"sig_vs_bkg_all_variables{s}", "title":f"N sig={len(sig_context)}, N bkg={len(bkg_context)} in SR", "outdir":args.outdir}
        plot_all_variables(sig_list, bkg_list, context_names+feature_names, **plot_kwargs)
        # data vs background
        plot_kwargs = {"labels":["data", "bkg"], "name":f"data_vs_bkg_all_variables{s}", "title":f"N sig={len(sig_context)}, N bkg={len(bkg_context)} in SR", "outdir":args.outdir}
        plot_all_variables(data_list, bkg_list, context_names+feature_names, **plot_kwargs)

        # Save dataset
        if args.supervised:
            np.savez(f"./{args.outdir}/supervised_inputs_{num}.npz", bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, bkg_mask_SR=bkg_mask_SR, sig_mask_SR = sig_mask_SR, sig_percent=s)
            
        else:
            np.savez(f"./{args.outdir}/inputs_s{num}.npz", data_feature=data_feature, data_context=data_context, MC_feature=MC_feature, MC_context=MC_context, bkg_feature=bkg_feature, bkg_context=bkg_context, ideal_bkg_feature=ideal_bkg_feature, ideal_bkg_context=ideal_bkg_context, sig_feature = sig_feature, sig_context = sig_context, data_mask_CR=data_mask_CR, data_mask_SR=data_mask_SR, MC_mask_CR=MC_mask_CR, MC_mask_SR=MC_mask_SR, bkg_mask_CR=bkg_mask_CR, bkg_mask_SR=bkg_mask_SR, ideal_bkg_mask_CR=idea_bkg_mask_CR, ideal_bkg_mask_SR=ideal_bkg_mask_SR, sig_mask_CR = sig_mask_CR, sig_mask_SR = sig_mask_SR, sig_percent=s)
        
        num += 1
        
        
    print("Finished generating dataset.")

if __name__ == "__main__":
    main()