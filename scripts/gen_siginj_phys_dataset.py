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
    "-b",
    "--bkgsample",
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
    '-t', 
    "--test",
    action="store_true",
    default=False,
    help='Generate test datasets.'
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
    

def main():

    # Create the output directory
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # Load input events ordered by varibles
    variables, sig = load_samples(args.sigsample)
    _, bkg = load_samples(args.bkgsample)
    _, MC = load_samples(args.mcsample)

    sig = get_quality_events(sig)
    bkg = get_quality_events(bkg)
    MC = get_quality_events(MC)
    
    # Reshape MC to match bkg
    n_MC = bkg.shape[0]
    selected_MC_indices = np.random.choice(MC.shape[0], size=n_MC, replace=False)
    MC = MC[selected_MC_indices, :] 
    
    # Prepare Bkg and MC
    
    # Create context array
    context_names = ["ht", "met"]
    bkg_context = sort_event_arr(context_names, variables, bkg)
    MC_context = sort_event_arr(context_names, variables, MC)
    
    # Create feature array
    feature_names = ["m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    bkg_feature = sort_event_arr(feature_names, variables, bkg)
    MC_feature = sort_event_arr(feature_names, variables, MC)
    
    # Define SR and CR masks
    HT_cut = 800    # In SR, HT > 800 GeV
    MET_cut = 75    # In SR, MET > 75 GeV
    
    MC_mask_SR = (MC_context[:, 0] > HT_cut) & (MC_context[:, 1] > MET_cut)
    MC_mask_CR = np.logical_not((MC_context[:, 0] > HT_cut) & (MC_context[:, 1] > MET_cut))
    
    bkg_mask_SR = (bkg_context[:, 0] > HT_cut) & (bkg_context[:, 1] > MET_cut)
    bkg_mask_CR = np.logical_not((bkg_context[:, 0] > HT_cut) & (bkg_context[:, 1] > MET_cut))
    
    
    # initialize lists
    if args.test:
        sig_percent_list = [1]
    elif args.supervised:
        sig_percent_list = [1]*20
    else:
        # sig_percent_list = [len(sig_context)/len(bkg_context)]
        # sig_percent_list = np.logspace(np.log10(0.0005),np.log10(0.01),10).round(5)
        sig_percent_list = np.logspace(np.log10(0.001),np.log10(0.01),3).round(5)
    
    
    # Create signal injection dataset
    num = 0
    for s in sig_percent_list:
        
        # Subsample siganl set
        n_sig = int(s * bkg.shape[0])
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
        print(f"S/B={s}, N sig={len(sig_context)}, N bkg={len(bkg_context)}, N MC={len(MC_context)}, data_context: {data_context.shape}, data_feature: {data_feature.shape}")
        
        
        # Save dataset
        if args.test:
            np.savez(f"./{args.outdir}/test_inputs.npz", bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, bkg_mask_SR=bkg_mask_SR, sig_mask_SR = sig_mask_SR, sig_percent=s)
        
        elif args.supervised:
            np.savez(f"./{args.outdir}/supervised_inputs_{num}.npz", bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, bkg_mask_SR=bkg_mask_SR, sig_mask_SR = sig_mask_SR, sig_percent=s)
            
        else:
            np.savez(f"./{args.outdir}/inputs_s{num}.npz", data_feature=data_feature, data_context=data_context, MC_feature=MC_feature, MC_context=MC_context, bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, data_mask_CR=data_mask_CR, data_mask_SR=data_mask_SR, MC_mask_CR=MC_mask_CR, MC_mask_SR=MC_mask_SR, bkg_mask_CR=bkg_mask_CR, bkg_mask_SR=bkg_mask_SR, sig_mask_CR = sig_mask_CR, sig_mask_SR = sig_mask_SR, sig_percent=s)
        
        num += 1
        
        
    print("Finished generating dataset.")

if __name__ == "__main__":
    main()