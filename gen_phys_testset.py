import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import *
from helpers.process_data import *
from semivisible_jet.utils import *
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument( "-s", "--sigsample",help="Input signal .txt file",
   #default="/global/cfs/cdirs/m3246/kbai/HV_samples/analysis_input_testset/rinv13_pTmin200GeV.txt"
   #default="/global/cfs/cdirs/m3246/kbai/HV_samples/analysis_input_testset/rinv13_2TeV.txt"
   default="/global/cfs/cdirs/m3246/kbai/HV_samples/analysis_input_testset/rinv13_3TeV.txt"
)
parser.add_argument("-b","--bkg-dir",help="Input bkground folder",
    default="/global/cfs/cdirs/m3246/kbai/HV_samples/qcd_test_samples/" 
)
parser.add_argument( "-size", type=int,help="Number of bkg text files",default=30)
parser.add_argument("-o","--outdir",help="output directory")

args = parser.parse_args()

def main():

    # Create the output directory
    data_dir = f"{args.outdir}/data/"
    os.makedirs(data_dir, exist_ok=True)
    
    # define sample size as the number of files
    sample_size = args.size
    # load in the preprocessor 
    with open(f"{data_dir}/mc_scaler.pkl","rb") as f:
        print("Loading in trained minmax scaler.")
        scaler = pickle.load(f)
        
    print(f"Loading {sample_size} samples...")
    
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
            
    print("Done loading!")

    # concatenate all backgroud events
    bkg_events = np.concatenate(bkg_events_list)

    # SR bkg
    bkg_mask_SR = apply_SR_cuts(bkg_events)
    bkg_events_SR = bkg_events[bkg_mask_SR]
    
    sig_mask_SR = apply_SR_cuts(sig_events)
    sig_events_SR = sig_events[sig_mask_SR]
    
    print(bkg_events_SR.shape, sig_events_SR.shape)
    
    # Select test set
    n_test_sig = 50000
    n_test_bkg = 200000
    sig_test_SR = sig_events_SR[:n_test_sig]
    bkg_test_SR = bkg_events_SR[:n_test_bkg]
    
    # Select fully supervised set
    sig_fullsup_SR = sig_events_SR[n_test_sig:]
    bkg_fullsup_SR = bkg_events_SR[n_test_bkg:]
   
    print(f"Test dataset in SR: N sig={len(sig_test_SR)}, N bkg={len(bkg_test_SR)}")
    print(f"Fully supervised dataset in SR: N sig={len(sig_fullsup_SR)}, N bkg={len(bkg_fullsup_SR)}")

    # Plot varibles
    plot_dir = f"{data_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    sig_list = sig_events_SR.T
    bkg_list = bkg_events_SR.T
    plot_kwargs = {"labels":["sig", "bkg"], "name":f"sig_vs_bkg_test", "title":f"N sig={len(sig_test_SR)}, N bkg={len(bkg_test_SR)}", "outdir":plot_dir}
    plot_all_variables(sig_list, bkg_list, var_names, **plot_kwargs)

    # Save dataset
    np.savez(f"{data_dir}/test_SR.npz", bkg_events_SR=scaler.transform(bkg_test_SR), sig_events_SR=scaler.transform(sig_test_SR))
    np.savez(f"{data_dir}/fullsup_SR.npz", bkg_events_SR=scaler.transform(bkg_fullsup_SR), sig_events_SR=scaler.transform(sig_fullsup_SR))
    
        
    print(f"Finished generating datasets.")

if __name__ == "__main__":
    main()