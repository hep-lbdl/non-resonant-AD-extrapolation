import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import *
from helpers.process_data import *
from semivisible_jet.utils import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--sigsample",
    action="store",
    help="Input signal .txt file",
    default="/global/cfs/cdirs/m3246/kbai/HV_samples/sig_samples/rinv13_pTmin200GeV.txt"
)
parser.add_argument(
    "-b1",
    "--bkg-dir",
    action="store",
    help="Input bkground .txt files",
    default="/global/cfs/cdirs/m3246/kbai/HV_samples/qcd_data_samples/"
)
parser.add_argument(
    "-b2",
    "--ideal-bkg-dir",
    action="store",
    help="Input ideal bkground .txt files",
    default="/global/cfs/cdirs/m3246/kbai/HV_samples/qcd_idealAD_samples/"
)
parser.add_argument(
    "-mc",
    "--mc-dir",
    action="store",
    help="Input MC bkground .txt files",
    default="/global/cfs/cdirs/m3246/kbai/HV_samples/qcd_mc_samples/" 
)
parser.add_argument(
    "-size",
    action="store",
    type=int,
    help="Number of bkg text files",
    default=20
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

    # Create the output directory
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # define sample size as the number of files
    sample_size = args.size
    print(f"Loading {sample_size} samples...")

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
    
    print("Done loading!")

    # concatenate all background events
    bkg_events = np.concatenate(bkg_events_list)
    ideal_bkg_events = np.concatenate(ideal_bkg_events_list)
    mc_events = np.concatenate(mc_events_list)

    # SR masks
    bkg_mask_SR = phys_SR_mask(bkg_events)
    bkg_mask_CR = ~bkg_mask_SR
    
    mc_mask_SR = phys_SR_mask(mc_events)
    mc_mask_CR = ~mc_mask_SR
    
    ideal_bkg_mask_SR = phys_SR_mask(ideal_bkg_events)
    ideal_bkg_mask_CR = ~ideal_bkg_mask_SR
        
    # save out the events that don't change with signal injection
    
    # mc CR and SR
    np.savez(f"{args.outdir}/mc_events.npz", mc_events_cr=mc_events[mc_mask_CR], mc_events_sr=mc_events[mc_mask_SR])
    # ideal CR and SR
    np.savez(f"{args.outdir}/ideal_bkg_events.npz", ideal_bkg_events_cr=ideal_bkg_events[ideal_bkg_mask_CR], ideal_bkg_events_sr=ideal_bkg_events[ideal_bkg_mask_SR])
    
    # initialize lists
    sig_percent_list = [0, 0.0025, 0.005, 0.0075, 0.012, 0.016, 0.02]

    # Create signal injection dataset
    n_bkg_SR = bkg_events[bkg_mask_SR].shape[0]
    
    for s in sig_percent_list:
        
        # Subsample signal set
        n_sig = int(s * n_bkg_SR)
        selected_sig_indices = np.random.choice(sig_events.shape[0], size=n_sig, replace=False)
        selected_sig = sig_events[selected_sig_indices, :] 

        # Create data arrays
        data_events = np.concatenate([selected_sig, bkg_events])

        # SR masks
        data_mask_SR = phys_SR_mask(data_events)
        data_mask_CR = ~data_mask_SR
        selected_sig_mask_SR = phys_SR_mask(selected_sig)

        # SR events
        n_sig_SR = selected_sig[selected_sig_mask_SR].shape[0]
        s_SR = round(n_sig_SR/n_bkg_SR, 5)
        signif = round(n_sig_SR/np.sqrt(n_bkg_SR), 5)

        # Print dataset information
        print(f"S/B={s_SR} in SR, S/sqrt(B) = {signif}, N bkg SR: {n_bkg_SR:.1e}, N sig SR: {n_sig_SR}")
        
        # Plot varibles in the SR
        sig_list = selected_sig[selected_sig_mask_SR].T
        bkg_list = bkg_events[bkg_mask_SR].T
        data_list = data_events[data_mask_SR].T
        plot_dir = f"{outdir}/plots"
        os.makedirs(plot_dir, exist_ok=True)
        # Signal vs background
        plot_kwargs = {"name":f"sig_vs_bkg_SR_{s}", "title":"Signal vs background in SR", "outdir":plot_dir}
        plot_all_variables(sig_list, bkg_list, var_names, **plot_kwargs)
        # data vs background SR
        plot_kwargs = {"labels":["data", "bkg"], "name":f"full", "title":"Data vs background in SR", "outdir":plot_dir}
        plot_all_variables(data_list, bkg_list, var_names, **plot_kwargs)
        
        # Save dataset
        np.savez(f"{args.outdir}/data_{s}.npz", data_events_cr=data_events[data_mask_CR], data_events_sr=data_events[data_mask_SR], sig_percent=s_SR)
    
        
        
    print("Finished generating dataset.")

if __name__ == "__main__":
    main()