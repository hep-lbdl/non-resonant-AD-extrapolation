import numpy as np
import matplotlib.pyplot as plt


def name_map():
    return {
        "m_jj": "$m_{{\\rm jj}}$",
        "met": "MET",
        "ht": "HT",
        "pT_j1": "Leading jet $p_{{\\rm T}}$",
        "pT_j2": "Sub-leading jet $p_{{\\rm T}}$",
        "tau21_j1": "Leading jet $\\tau_2/\\tau_1$",
        "tau21_j2": "Sub-leading jet $\\tau_2/\\tau_1$",
        "tau32_j1": "Leading jet $\\tau_2/\\tau_2$",
        "tau32_j2": "Sub-leading jet $\\tau_3/\\tau_2$",
        "min_dPhi": "min$\\Delta\\phi(\\rm j_i, \\rm MET)$",
    }

def unit_map():
    return {
        "m_jj": "GeV",
        "met": "GeV",
        "ht": "GeV",
        "pT_j1": "GeV",
        "pT_j2": "GeV",
        "tau21_j1": "",
        "tau21_j2": "",
        "tau32_j1": "",
        "tau32_j2": "",
        "min_dPhi": "",
    }


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
        bins = np.linspace(np.min(data_list[0]), np.max(data_list[0]), 20)
    for i in range(len(label_list)):
        plt.hist(data_list[i], bins = bins, density = True, histtype='step', label=label_list[i])

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Events (A.U)", fontsize=14)
    plt.legend(fontsize=14)
    plt.show
    if len(figname)>0:
        plt.savefig(f"{outdir}/{figname}.png")
    plt.close

    
def plot_quantity_list_ratio(data_list, label_list, title, xlabel, bins=None, figname="", outdir="plots"):
    plt.figure(figsize=(8,6))

    fig, ax = plt.subplots(2, figsize = (8, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    if bins is None:
        bins = np.linspace(np.min(data_list[0]), np.max(data_list[0]), 20)
        
    for i in range(len(label_list)):
        ax[0].hist(data_list[i], bins = bins, density = True, histtype='step', label=label_list[i])
    
    mask = data_list[0] != 0
    ratio = data_list[0][mask]/data_list[1][mask]
    
    ax[1].hist(ratio, bins = bins, density = False, histtype='step', label=label_list[i])
    ax[1].set_xlabel(xlabel, fontsize=14)
    ax[1].set_ylim([0,2])
    
    ax[0].set_ylabel("Events (A.U)", fontsize=14)  
    ax[1].set_ylabel("sig/bkg")

    plt.title(title, fontsize=16)
    plt.legend(fontsize=14)
    plt.show
    if len(figname)>0:
        plt.savefig(f"{outdir}/{figname}_ratio.png")
    plt.close
    
