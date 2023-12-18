import numpy as np
import sys
import os

# Assume all events has the structure of the following:
# events[:, 0] is context variable 1
# events[:, 1] is context variable 2
# events[:, 1:] are the feature variables

def get_context(events):

    if events.shape[1]>1:
        return events[:, :1]
    else:
        sys.exit(f"Wrong input events array. Array dim {events.shape[1]}, must be >= 2. Exiting...")


def get_feature(events):

    if events.shape[1]>1:
        return events[:, 1:]
    else:
        sys.exit(f"Wrong input events array. Array dim {events.shape[1]}, must be >= 2. Exiting...")


def toy_SR_mask(events):

    # define SR and CR masks
    m1_cut = 1    # In SR, m1 > 1
    m2_cut = 1    # In SR, m2 > 1
    
    # SR masks
    if events.shape[1]>1:
        mask_SR = (events[:, 0] > m1_cut) & (events[:, 1] > m2_cut)
        return mask_SR
    else:
        sys.exit(f"Wrong input events array. Array dim {events.shape[1]}, must be >= 2. Exiting...")

def phys_SR_mask(events):

    # define SR and CR masks
    HT_cut = 800    # In SR, HT > 800 GeV
    MET_cut = 75    # In SR, MET > 75 GeV

    # SR masks
    if events.shape[1]>1:
        mask_SR = (events[:, 0] > HT_cut) & (events[:, 1] > MET_cut)
        return mask_SR
    else:
        sys.exit(f"Wrong input events array. Array dim {events.shape[1]}, must be >= 2. Exiting...")
        

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

def morph_mc(mc_events):
    """
    This function has been hand-tuned to samples in the official Zenodo dataset. Be aware!!
    """
    morphed_mc_events = np.copy(mc_events)

    def morph_ht(x):
        return x

    def morph_met(x):
        return (x*(1+(x/500.)))

    def morph_mjj(x):
        return x*(1+(x/6000.))

    def morph_taus(x):
        return x*(x**(0.1))

    morphed_mc_events[:,0] = morph_ht(morphed_mc_events[:,0])
    morphed_mc_events[:,1] = morph_met(morphed_mc_events[:,1])
    morphed_mc_events[:,2] = morph_mjj(morphed_mc_events[:,2])
    morphed_mc_events[:,3] = morph_taus(morphed_mc_events[:,3])
    morphed_mc_events[:,4] = morph_taus(morphed_mc_events[:,4])
    morphed_mc_events[:,5] = morph_taus(morphed_mc_events[:,5])
    morphed_mc_events[:,6] = morph_taus(morphed_mc_events[:,6])

    return morphed_mc_events