import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys

def name_map():
    return {
        "m_jj": "$m_{{\\rm jj}}$",
        "met": "MET",
        "ht": "$H_{{\\rm T}}$",
        "pT_j1": "Leading jet $p_{{\\rm T}}$",
        "pT_j2": "Sub-leading jet $p_{{\\rm T}}$",
        "tau21_j1": "Leading jet $\\tau_2/\\tau_1$",
        "tau21_j2": "Sub-leading jet $\\tau_2/\\tau_1$",
        "tau32_j1": "Leading jet $\\tau_3/\\tau_2$",
        "tau32_j2": "Sub-leading jet $\\tau_3/\\tau_2$",
        "min_dPhi": "min$\\Delta\\phi(\\rm j_i, \\rm MET)$",
    }

def unit_map():
    return {
        "m_jj": "(GeV)",
        "met": "(GeV)",
        "ht": "(GeV)",
        "pT_j1": "(GeV)",
        "pT_j2": "(GeV)",
        "tau21_j1": "",
        "tau21_j2": "",
        "tau32_j1": "",
        "tau32_j2": "",
        "min_dPhi": "",
    }

def load_samples(file):
    samples = np.loadtxt(file, dtype=str)
    # Get the names of all varibles
    variables = samples[0]
    # Get the events ordered by varibles
    events = np.asfarray(samples[1:])
    return variables, events


def sort_event_arr(names, variables, events):
    
    event_list = []
    
    for x in names:   
        ind_x = ind(variables, x)
        event_list.append(events[:, ind_x])
    
    return np.stack(event_list, axis=1)


def apply_SR_cuts(events):

    # define SR and CR masks
    HT_cut = 600    # In SR, HT > 600 GeV
    MET_cut = 75    # In SR, MET > 75 GeV
    
    # SR masks
    if events.shape[1]>1:
        mask_SR = (events[:, 0] > HT_cut) & (events[:, 1] > MET_cut)
        return mask_SR
    else:
        sys.exit(f"Wrong input events array. Array dim {events.shape[1]}, must be >= 2. Exiting...")


def ind(variables, name):
    return np.where(variables == name)[0][0]

def plot_quantity(data, label, title, xlabel, figname=""):
    plt.figure(figsize=(8,6))
    bins = np.linspace(np.min(data), np.max(data), 20)
    plt.hist(data, bins = bins, density = True, histtype='step', label=label)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Events (A.U)", fontsize=14)
    plt.legend(fontsize=14)
    plt.show
    if len(figname)>0:
        plt.savefig(f"plots/{figname}.png")
    plt.close
    
def plot_quantity_list(data_list, label_list, title, xlabel, bins=None, figname="", outdir="plots"):
    plt.figure(figsize=(8,6))
    if bins is None:
        bins = np.linspace(np.min(data_list[0]), np.max(data_list[0]), 30)
    
    for i in range(len(label_list)):
        if i == len(label_list)-2:
            plt.hist(data_list[i], bins = bins, density = True, ls='--', color='darkred', histtype='step', label=label_list[i])
        elif i == len(label_list)-1:
            plt.hist(data_list[i], bins = bins, density = True, ls='--', color='darkslategrey', histtype='step', label=label_list[i])
        else:
            plt.hist(data_list[i], bins = bins, density = True, histtype='step', label=label_list[i])

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Events (A.U.)", fontsize=14)
    plt.legend(fontsize=14)
    plt.show
    if len(figname)>0:
        plt.savefig(f"{outdir}/{figname}.png")
    plt.close
    plt.clf()

    
def plot_quantity_list_ratio(data_list, label_list, title, xlabel, bins=None, figname="", outdir="plots"):

    fig, ax = plt.subplots(2, figsize = (6, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    if bins is None:
        lb = np.min(data_list[1])
        ub = np.max(data_list[0])
        bins = np.linspace(lb, ub, 30)
    
    c_list = []
    for i in range(len(label_list)):
        c0, cbins, _ = ax[0].hist(data_list[i], bins = bins, density = True, histtype='step', label=label_list[i])
        c_list.append(c0)
    
    r_c = [0.0]
    r_s = [0.0]
    
    for i in range(len(bins)-1):
        # r_c = S/B in CR
        sig_c = c_list[0][:i+1]
        bkg_c = c_list[1][:i+1]
        
        if np.sum(bkg_c) != 0:
            r_c.append(np.sum(sig_c)/np.sum(bkg_c))
        else:
            r_c.append(np.nan)

        # r_s = S/B in SR
        sig_s = c_list[0][i+1:]
        bkg_s = c_list[1][i+1:]
        
        if np.sum(bkg_s) != 0:
            r_s.append(np.sum(sig_s)/np.sum(bkg_s))
        else:
            r_s.append(np.nan)

    # Ensure bins and ratio are numpy arrays
    bins = np.array(bins)
    r_c = np.array(r_c)
    r_s = np.array(r_s)
    
    # r_c/r_s
    r_cs = np.divide(r_c, r_s, out=np.full_like(r_c, np.nan), where=(r_s != 0))
    
    ax[1].plot(bins, r_s, color='green', marker='.', label=r"$r_{\rm SR}$ = S/B in SR")
    ax[1].plot(bins, r_c, color='blue', marker='.', label=r"$r_{\rm CR}$ = S/B in CR")
    ax[1].plot(bins, r_cs, color='orange', marker='.', label=r"$r_{\rm CR}/r_{\rm SR}$")
    ax[1].set_xlabel(xlabel, fontsize=14)
    ax[1].set_yscale('log')
    
    ax[0].set_xticks([])
    ax[0].set_ylabel("Events", fontsize=12)  
    ax[1].set_ylabel("ratio")
    
    # ax[0].legend(fontsize=12)
    ax[0].set_title(title, fontsize=14)
    
    # set legend position
    fig.legend(bbox_to_anchor=(0.9, 0.85), fontsize=10)
    
    plt.show
    if len(figname)>0:
        plt.savefig(f"{outdir}/{figname}_ratio.png")

    plt.clf()
    plt.close(fig)
    

def plot_correlation_hist(x, y, xlabel, ylabel, xmax, ymax, title, figname="", outdir="plots"):
    
    plt.figure(figsize=(6,6))
    plt.hist2d(x, y, bins=40, range=[[0, xmax], [0, ymax]], norm=colors.LogNorm())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show
    plt.savefig(f'{outdir}/correlation{figname}.png')
    plt.close
    

def plot_CR_SR(x, y, xlabel, ylabel, title, figname=""):
    
    plt.figure(figsize=(6,6))
    plt.scatter(x1, x2, alpha = 0.2, color="blue", label = 'bkg')
    plt.scatter(y1, y2, alpha = 0.2, color = "organe", label = 'sig')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.axvline(x=1, color='r', linestyle='-')
    plt.legend()
    plt.title("Control region and signal region")
    plt.show
    plt.savefig('plots/CR_SR_bkg_sig.png')
    plt.close