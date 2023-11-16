import numpy as np
from math import pi
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.special import kl_div
from datetime import datetime
import os

os.makedirs(os.path.dirname("./plots"), exist_ok=True)

def get_kl_div(p,q):
    div_arr = np.where(np.logical_and(p>0,q>0),kl_div(p,q),0)
    return np.sum(div_arr)
    

def plot_kl_div_toy(x1, x2, label1, label2, w1=None, w2=None, name="feature", title = "", bins=50, outdir="plots", *args, **kwargs):
    
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    fig, ax1 = plt.subplots(figsize=(8,6))
    c0, cbins, _ = ax1.hist(x1, bins = bins, density = True, weights=w1, histtype='step', color=colors[2], label=label1)
    c1, _, _ = ax1.hist(x2, bins = cbins, density = True, weights=w2, histtype='stepfilled', alpha = 0.3, color=colors[2], label=label2)
    kl_div = get_kl_div(c0,c1)
    ax1.hist(x2, bins = bins, density = True, histtype='stepfilled', alpha = 0, color=colors[2], label=f"kl div={kl_div:.3f}")
    ax1.set_title(title, fontsize = 14)
    ax1.set_xlabel(name)
    plt.legend(fontsize = 10)
    plt.show
    plot_name = f"{outdir}/{label1}_{label2}_{name}.png"
    plt.savefig(plot_name.replace(" ", "_"))
    plt.close()
        
        

def plot_kl_div_data_reweight(data_train, data_true, data_gen, weights, data_gen_from_truth=None, MC_true=None, name="data_reweight", title="", ymin=-6, ymax=10, outdir="./", *args, **kwargs):
    
    # data_train is the data in CR
    # data_true is the data in SR
    # data_gen is the predicted data in SR
    # weights is used to reweight MC to data
    # 
    # optional:
    # MC_true
    # data_gen_from_truth
    
    
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    bins = np.linspace(ymin, ymax, 50)
    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.hist(data_train, bins = bins, density = True, histtype='step', ls="--", color=colors[0], label=f"data in CR")
    
    c0, cbins, _ = ax1.hist(data_true, bins = bins, density = True, histtype='step', color=colors[0], label=f"data in SR")
    
    if MC_true is not None:
        ax1.hist(MC_true, bins = bins, density = True, histtype='step', color=colors[1], label=f"MC in SR")
    
    ax1.hist(data_gen, bins = bins, density = True, histtype='stepfilled', alpha = 0.3, color=colors[1], label=f"no weight pred bkg in SR from MC")
    
    c1, cbins, _ = ax1.hist(data_gen, bins = bins, density = True, weights=weights, histtype='stepfilled', alpha = 0, color=colors[0])
    kl_div = get_kl_div(c0,c1)
    ax1.hist(data_gen, bins = bins, density = True, weights=weights, histtype='stepfilled', alpha = 0.3, color=colors[0], label=f"pred bkg in SR from MC (kl div from data = {kl_div:.3f})")
    
    if data_gen_from_truth is not None:
        c2, cbins, _ = ax1.hist(data_gen_from_truth, bins = bins, density = True, histtype='stepfilled', alpha = 0, color=colors[3])
        kl_div = get_kl_div(c0,c2)
        ax1.hist(data_gen_from_truth, bins = bins, density = True, histtype='stepfilled', alpha = 0.3, color=colors[3], label=f"pred bkg in SR from data (kl div from data = {kl_div:.3f})")
    
    ax1.set_title(f"True vs predicted background in SR {title}", fontsize = 14)
    ax1.set_xlabel("x")
    plt.legend(loc='upper left', fontsize = 9)
    plt.show
    plot_name = f"{outdir}/{name}.png"
    plt.savefig(plot_name.replace(" ", "_"))
    plt.close()
  
    
    
def plot_multi_dist(hists, labels, weights=None, htype=None, lstyle=None, title="", name="", xlabel="x", ymin=-10, ymax=10, outdir="./", *args, **kwargs):
    colors = ['royalblue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    alphas = [1, 0.2, 1, 1, 0.3, 1, 1, 1]
    # colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    N = len(hists)
    
    if N==len(hists) and N==len(labels) and N<=len(colors):
        bins = np.linspace(ymin, ymax, 50)
        fig, ax = plt.subplots(2, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1]})
        c_list = []
        for i in range(3):
            
            w_i = weights[i] if weights is not None else None
            ht_i = htype[i] if htype is not None else 'step'
            ls_i = lstyle[i] if lstyle is not None else '-'
                
            c, cbins, _ = ax[0].hist(hists[i], bins = bins, lw=2, density = True, weights=w_i, histtype=ht_i, ls=ls_i, alpha=alphas[i], color=colors[i], label=f"{labels[i]}")
            c_list.append(c)

        # ratio of weighted SR vs true bkg SR
        MC_bkg_SR = np.array(c_list[0])
        weighted_bkg_SR = np.array(c_list[1])
        target_bkg_SR = np.array(c_list[2])
        r_mcbkg = np.divide(MC_bkg_SR, target_bkg_SR, out=np.full_like(MC_bkg_SR, np.nan), where=(target_bkg_SR != 0))
        r_bkg = np.divide(weighted_bkg_SR, target_bkg_SR, out=np.full_like(weighted_bkg_SR, np.nan), where=(target_bkg_SR != 0))
        
        # plot ratio
        ax[1].plot(cbins[:-1], r_bkg, color='slategrey', marker='.', lw=2)
        # ax[1].plot(cbins[:-1], r_mcbkg, color='royalblue', marker='.', lw=2, alpha=0.6)
        ax[1].axhline(y=1, color='black', linestyle='-')
 
        ax[1].set_xlabel(xlabel, fontsize=14)
        ax[1].set_ylabel("Ratio to truth", fontsize=16)
        ax[1].set_xlabel(xlabel, fontsize=14)
        ax[1].set_ylim(0.5, 1.5)
        ax[1].tick_params(axis='both', which='major', labelsize=12)

        ax[0].set_ylabel("Events (a.u.)", fontsize=16)
        ax[0].set_xticks([]) 
        # ax[0].set_yticks([])
        # plt.legend(loc='upper right', fontsize = 14)
        # set legend position
        ax[0].legend(fontsize=16)

        plot_name = f"{outdir}/{name}.png"
        plot_name = plot_name.replace(" ", "_")
        plt.savefig(plot_name)
        plt.close()
        print(f"Reweighting plot saved as {plot_name}")
    else:
        print("Wrong input lists!")


def plot_kl_div_phys(x1, x2, label1, label2, w1=None, w2=None, name="feature", tag = "", bins=50, outdir="plots", *args, **kwargs):
    
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    fig, ax = plt.subplots(2, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1]})

    c0, cbins, _ = ax[0].hist(x1, bins = bins, density = True, weights=w1, histtype='step', lw=2, color=colors[2], label=label1)
    c1, cbins, _ = ax[0].hist(x2, bins = bins, density = True, weights=w2, histtype='stepfilled', alpha = 0.2, color=colors[1], label=label2)

    # ratio of gen SR vs true SR
    gen_bkg_SR = np.array(c1)
    target_bkg_SR = np.array(c0)
    r_bkg = np.divide(gen_bkg_SR, target_bkg_SR, out=np.full_like(gen_bkg_SR, np.nan), where=(target_bkg_SR != 0))

    # plot ratio
    ax[1].plot(cbins[:-1], r_bkg, color='slategrey', marker='.', lw=2)
    ax[1].axhline(y=1, color='black', linestyle='-')
    ax[1].set_xlabel(name, fontsize=14)
    ax[1].set_ylabel("Ratio to truth", fontsize=16)
    ax[1].set_ylim(0.5, 1.5)
    ax[1].tick_params(axis='both', which='major', labelsize=12)

    ax[0].set_ylabel("Events (a.u.)", fontsize=14)
    ax[0].set_xticks([]) 
    ax[0].legend(fontsize=16)
    
    plot_name = f"{outdir}/{label1}_{label2}_{tag}.png"
    plot_name = plot_name.replace(" ", "_")
    plt.savefig(plot_name)
    print(f"MAF plots saved as {plot_name}")
    plt.close()

def plot_multi_data_MC_dist(data_list, MC_list, labels, weights=None, name="data_vs_mc", title="", xlabel="x", ymin=-10, ymax=10, outdir="./", *args, **kwargs):
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    colors = colors + colors
    
    N = len(data_list)
    
    if N==len(MC_list) and N==len(labels) and N<=len(colors):
        bins = np.linspace(ymin, ymax, 50)
        fig, ax1 = plt.subplots(figsize=(10,6))
        for i in range(N):
            if weights is not None:
                w_i = weights[i]
            else:
                w_i = None
            ax1.hist(data_list[i], bins = bins, density = True, histtype='step', ls= "-", color=colors[i], label=f"data {labels[i]}")
            ax1.hist(MC_list[i], bins = bins, density = True, weights=w_i, histtype='step', ls= "--", color=colors[i], label=f"MC {labels[i]}")
        ax1.set_title(f"{title}", fontsize = 14)
        ax1.set_xlabel(xlabel)
        plt.legend(loc='upper left', fontsize = 9)
        plt.show
        plot_name = f"{outdir}/{name}.png"
        plt.savefig(plot_name.replace(" ", "_"))
        plt.close()
    else:
        print("Wrong input lists!")
        
        
def plot_sig_bkg_dist(sig_list, bkg_hist, labels, name="sig_vs_bkg", title="", xlabel="x", ymin=-10, ymax=10, outdir="./", *args, **kwargs):
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    colors = colors + colors
    
    N = len(sig_list)
    
    if N==len(labels) and N<=len(colors):
        bins = np.linspace(ymin, ymax, 50)
        fig, ax1 = plt.subplots(figsize=(10,6))
        
        ax1.hist(bkg_hist, bins = bins, density = True, histtype='step', ls= "--", color=colors[-1], label=f"Bkg")
        
        for i in range(N):
            ax1.hist(sig_list[i], bins = bins, density = True, histtype='step', ls= "-", color=colors[i], label=f"sig {labels[i]}")
        
        ax1.axvline(x=1, color='red', linestyle='--', label=f'{xlabel}=1')
        ax1.set_title(f"{title}", fontsize = 14)
        ax1.set_xlabel(xlabel)
        plt.legend(loc='upper left', fontsize = 9)
        plt.show
        plot_name = f"{outdir}/{name}.png"
        plt.savefig(plot_name.replace(" ", "_"))
        plt.close()
    else:
        print("Wrong input lists!")

def plot_all_variables(sig_list, bkg_list, xlabels, labels=["sig", "bkg"], name="sig_vs_bkg", title="", outdir="./", *args, **kwargs):
    csig = 'brown'
    cbkg = 'royalblue'
    
    N = len(sig_list)
    
    if N==len(xlabels):
        fig, ax1 = plt.subplots(1, N, figsize=(6*N,5))
        ax1[0].set_ylabel("Events (A.U.)")
        for i in range(N):
            xmin = np.min(np.hstack([bkg_list[i], sig_list[i]]))
            xmax = np.max(np.hstack([bkg_list[i], sig_list[i]]))
            bins = np.linspace(xmin, xmax, 50)
            ax1[i].hist(sig_list[i], bins = bins, density = False, histtype='step', ls= "-", color=csig, label=labels[0])
            ax1[i].hist(bkg_list[i], bins = bins, density = False, histtype='stepfilled', ls= "-", color=cbkg, alpha=0.5, label=labels[1])
            ax1[i].set_xlabel(xlabels[i])
            ax1[i].set_yticks([])
            ax1[i].legend(loc='upper right', fontsize = 9)

        plt.show
        plt.title(title)
        plot_name = f"{outdir}/{name}.png"
        plt.savefig(plot_name.replace(" ", "_"))
        plt.close()
    else:
        print("Wrong input lists!")
        
def plot_SIC(tpr, fpr, label, outdir="./"):
    
    tpr = np.array(tpr)
    fpr = np.array(fpr)

    SIC = tpr[fpr>0] / np.sqrt(fpr[fpr>0])
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(tpr[fpr>0], SIC, label=f"{label}")
    ax.set_ylabel(r"SIC = $\frac{\rm TPR}{\sqrt{\rm FPR}}$")
    ax.set_xlabel("Signal Efficiency (TPR)")
    ax.set_title(f"Significant improvement characteristic")
    # ax.plot([0,1],[0,1],color="gray",ls=":",label="Random")
    timestamp = datetime.now().strftime("%m-%d-%H%M%S")
    fname = f"{outdir}/SIC_{timestamp}.png"
    ax.legend()
    fig.savefig(fname)
    plt.close()
    
def plot_SIC_lists(tpr_list, fpr_list, sig_percent_list, name="", outdir="./"):
    
    label_list = [f"S/B={percent*100:.3f}%" for percent in sig_percent_list]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    for i in range(len(tpr_list)):
    
        tpr = np.array(tpr_list[i])
        fpr = np.array(fpr_list[i])

        SIC = tpr[fpr>0] / np.sqrt(fpr[fpr>0])

        ax.plot(tpr[fpr>0], SIC, label=f"{label_list[i]}")
        ax.set_ylabel(r"SIC = $\frac{\rm TPR}{\sqrt{\rm FPR}}$", fontsize=14)
        ax.set_xlabel("Signal Efficiency (TPR)", fontsize=14)
        ax.set_title(f"Significant improvement characteristic {name}", fontsize=14)
        # ax.plot([0,1],[0,1],color="gray",ls=":",label="Random")
    fname = f"{outdir}/SIC_sig_inj.png"
    ax.legend()
    fig.savefig(fname)
    plt.close()

def plot_rej_lists(tpr_list, fpr_list, sig_percent_list, name="", outdir="./"):
    
    label_list = [f"S/B={percent*100:.3f}%" for percent in sig_percent_list]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    for i in range(len(tpr_list)):
    
        tpr = np.array(tpr_list[i])
        fpr = np.array(fpr_list[i])

        rej = 1/fpr[fpr>0]

        ax.plot(tpr[fpr>0], rej, label=f"{label_list[i]}")
        ax.set_yscale('log')
        ax.set_ylabel(r"rejection = $\frac{1}{\rm FPR}$", fontsize=14)
        ax.set_xlabel("Signal Efficiency (TPR)", fontsize=14)
        ax.set_title(f"Rejection {name}", fontsize=14)
        # ax.plot([0,1],[0,1],color="gray",ls=":",label="Random")
    fname = f"{outdir}/rejection_sig_inj.png"
    ax.legend()
    fig.savefig(fname)
    plt.close()

    
def plot_max_SIC(sig_percent, max_SIC, label="", outdir="./"):
    
    sig_percent = np.array(sig_percent)*100
    
    plt.figure(figsize=(7, 5))
    plt.plot(sig_percent, max_SIC, '-', label=label)  # Line color (default)
    plt.plot(sig_percent, max_SIC, 'x', color='black')  # Marker color (black)
    # plt.xscale('log')
    plt.ylabel(r"max SIC")
    plt.xlabel("S/B (%)")
    plt.legend()
    plt.title("max SIC per signal significance")
    plt.savefig(f"{outdir}/maxSIC_sig_inj.png")
    plt.close()
    

def plot_multi_max_SIC(sig_percent, max_SIC_list, label_list, outdir="./"):
    
    sig_percent = np.array(sig_percent)*100
    
    plt.figure(figsize=(7, 5))
    
    for i in range(len(max_SIC_list)):
    
        plt.plot(sig_percent, max_SIC_list[i], '-', label=label_list[i])  # Line color (default)
        plt.plot(sig_percent, max_SIC_list[i], 'x', color='black')  # Marker color (black)
    
    plt.xscale('log')
    plt.ylabel(r"max SIC")
    plt.xlabel("S/B (%)")
    plt.legend()
    plt.title(f"Maximum significance improvement of each method")
    plt.savefig(f"{outdir}/maxSIC_sig_inj.png")
    plt.close()
    

def plot_avg_max_SIC(sig_percent, max_SIC_list, lb_list, ub_list, label_list, outdir="./", title="Maximum significance improvement", tag=""):
    
    colors = ['teal', 'royalblue', 'limegreen', 'darkorange', 'mediumorchid']
    
    sig_percent = np.array(sig_percent)*100
    
    plt.figure(figsize=(7, 5))
    
    for i in range(len(max_SIC_list)):

        plt.plot(sig_percent, max_SIC_list[i], '-x', label=label_list[i], color=colors[i])
        plt.fill_between(sig_percent, lb_list[i], ub_list[i], alpha = 0.2, color = colors[i])
    
    plt.xscale('log')
    plt.ylabel(r"max SIC")
    plt.xlabel("S/B (%)")
    plt.legend(edgecolor='none', facecolor='none')
    plt.title(f"{title}")
    plt.savefig(f"{outdir}/avg_maxSIC{tag}.png")
    plt.close()
    