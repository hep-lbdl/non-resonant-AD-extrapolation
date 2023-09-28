import numpy as np
from math import pi
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.special import kl_div
import os

os.makedirs(os.path.dirname("./plots"), exist_ok=True)

def get_kl_div(p,q):
    div_arr = np.where(np.logical_and(p>0,q>0),kl_div(p,q),0)
    return np.sum(div_arr)

def plot_gen_full_bkg(samples, x1, x2):
    y1 = samples[:,0]
    y2 = samples[:,1]
    plt.figure(figsize=(6,6))
    plt.scatter(x1, x2, alpha = 0.2, label = 'true bkg')
    plt.scatter(y1, y2, alpha = 0.2, color = "lightgreen", label = 'generated bkg')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.axvline(x=1, color='r', linestyle='-')
    plt.legend()
    plt.title("Generated full bkg distribution from training full bkg samples.")
    plt.show
    plt.savefig('plots/gen_full_bkg.pdf')
    plt.close
    
def plot_gen_SRfromCR_bkg(samples, x1, x2):
    y1 = samples[:,0]
    y2 = samples[:,1]
    plt.figure(figsize=(6,6))
    plt.scatter(x1, x2, alpha = 0.2, label = 'true bkg')
    plt.scatter(y1, y2, alpha = 0.2, color = "lightgreen", label = 'generated bkg in SR')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.axvline(x=1, color='r', linestyle='-')
    plt.legend()
    plt.title("Generated full bkg distribution from training CR samples.")
    plt.show
    plt.savefig('plots/gen_SRfromCR_bkg.pdf')
    plt.close
    
def plot_gen_SR_bkg_in_y_random(samples, Y_SR):
    samples = np.array(samples)
    plt.figure(figsize=(6,4))
    bins = np.linspace(-5, 5, 50)
    plt.hist(samples, bins = bins, density = True, histtype='step', label='generated SR')
    plt.hist(Y_SR, bins = bins, density = True, histtype='step', label='true SR')
    plt.title("Generated bkg distribution in x = random.")
    plt.legend()
    plt.show
    plt.savefig('plots/gen_SR_bkg_in_y_random.pdf')
    plt.close
    
def plot_gen_SR_bkg_in_y_cond(samples, Y_SR, k, q):
    samples = np.array(samples)
    bins = np.linspace(-5, 5, 50)
    plt.figure(figsize=(6,4))
    plt.hist(samples, bins = bins, density = True, histtype='step', label='generated SR')
    plt.hist(Y_SR, bins = bins, density = True, histtype='step', label='true SR')
    plt.title(f"Generated bkg in x = N(${k}\\alpha$+{q}$\\beta$, 1)")
    plt.legend()
    plt.show
    plt.savefig(f'plots/gen_SR_bkg_in_y_cond{k*10}.pdf')
    plt.close

def pi_to_string(theta):
    return f"({str(Fraction(theta/pi))})$\\pi$"
    
def plot_kl_div(Y_list, Y_list2, Y_label, Y_label2, k, theta=None, weights1=None, weights2=None, title="Normal($k(cos\\theta\\alpha + sin\\theta\\beta)$, 1)", tag = "", ymin=-6, ymax=10, outdir="plots", *args, **kwargs):
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    N = len(Y_list)
    
    if N==len(Y_list2) and N==len(k) and N<=len(colors):
        bins = np.linspace(ymin, ymax, 50)
        fig, ax1 = plt.subplots(figsize=(10,6))
        
        for i in range(N):
            
            if theta is None:
                label_k=f"k={k[i]}"
            elif N==len(theta):
                label_k=f"k={k[i]}, $\\theta$={pi_to_string(theta[i])}"
            else:
                print("Wrong theta lists!")
                break
            
            if weights1 is not None:
                w1_i = weights1[i]
            else:
                w1_i = None
                
            if weights2 is not None:
                w2_i = weights2[i]
            else:
                w2_i = None
            
            c0, cbins, _ = ax1.hist(Y_list[i], bins = bins, density = True, weights=w1_i, histtype='step', color=colors[i], label=f"{Y_label}, {label_k}")
            c1, cbins, _ = ax1.hist(Y_list2[i], bins = bins, density = True, weights=w2_i, histtype='stepfilled', alpha = 0.3, color=colors[i], label=f"{Y_label2}, {label_k}")
            kl_div = get_kl_div(c0,c1)
            ax1.hist(Y_list2[i], bins = bins, density = True, histtype='stepfilled', alpha = 0, color=colors[i], label=f"kl div={kl_div:.3f}")
        ax1.set_title(f"Background in x = {title}", fontsize = 14)
        ax1.set_xlabel("x")
        plt.legend(loc='upper left', fontsize = 9)
        plt.show
        plot_name = f"{outdir}/{Y_label}_{Y_label2}_{tag}.pdf"
        plt.savefig(plot_name.replace(" ", "_"))
        plt.close()
    else:
        print("Wrong input lists!")

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
    plot_name = f"{outdir}/{name}.pdf"
    plt.savefig(plot_name.replace(" ", "_"))
    plt.close()
  
    
def plot_results(k_list, theta_list, Y_list, samples_CR_list, samples_SR_list, mask_CR, mask_SR, plot_kwargs):
    Y_SR_list = []
    Y_CR_list = []
    Y_gen_CR_list = []
    Y_gen_SR_list = []

    for i in range(len(k_list)):
        Y_SR_list.append(Y_list[i][mask_SR])
        Y_CR_list.append(Y_list[i][mask_CR])
        Y_gen_CR_list.append(samples_CR_list[i])
        Y_gen_SR_list.append(samples_SR_list[i])

    plot_kl_div(Y_SR_list, Y_CR_list, "true SR", "true CR", k_list, theta_list, **plot_kwargs)
    plot_kl_div(Y_CR_list, Y_gen_CR_list, "true CR", "gen CR", k_list, theta_list, **plot_kwargs)
    plot_kl_div(Y_SR_list, Y_gen_SR_list, "true SR", "gen SR", k_list, theta_list, **plot_kwargs)
    
    
def plot_multi_dist(hists, labels, weights=None, htype=None, lstyle=None, title="", name="", xlabel="x", ymin=-10, ymax=10, outdir="./", *args, **kwargs):
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    alphas = [1, 0.2, 1, 1, 0.3, 1, 1, 1]
    # colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    N = len(hists)
    
    if N==len(hists) and N==len(labels) and N<=len(colors):
        bins = np.linspace(ymin, ymax, 50)
        fig, ax1 = plt.subplots(figsize=(10,6))
        for i in range(N):
            
            w_i = weights[i] if weights is not None else None
            ht_i = htype[i] if htype is not None else 'step'
            ls_i = lstyle[i] if lstyle is not None else '-'
                
            ax1.hist(hists[i], bins = bins, density = True, weights=w_i, histtype=ht_i, ls=ls_i, alpha=alphas[i], color=colors[i], label=f"{labels[i]}")
            
        ax1.set_title(f"{title}", fontsize = 14)
        ax1.set_xlabel(xlabel)
        plt.legend(loc='upper left', fontsize = 9)
        plt.show
        plot_name = f"{outdir}/{name}.pdf"
        plt.savefig(plot_name.replace(" ", "_"))
        plt.close()
    else:
        print("Wrong input lists!")
        
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
        plot_name = f"{outdir}/{name}.pdf"
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
        plot_name = f"{outdir}/{name}.pdf"
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
    fname = f"{outdir}/SIC.png"
    ax.legend()
    fig.savefig(fname)
    
def plot_SIC_lists(tpr_list, fpr_list, sig_percent_list, name="", outdir="./"):
    
    label_list = [f"S/B={percent*100:.3f}%" for percent in sig_percent_list]
    max_SIC_list = []
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    for i in range(len(tpr_list)):
    
        tpr = np.array(tpr_list[i])
        fpr = np.array(fpr_list[i])

        SIC = tpr[fpr>0] / np.sqrt(fpr[fpr>0])
        max_SIC_list.append(np.max(SIC))

        ax.plot(tpr[fpr>0], SIC, label=f"{label_list[i]}")
        ax.set_ylabel(r"SIC = $\frac{\rm TPR}{\sqrt{\rm FPR}}$")
        ax.set_xlabel("Signal Efficiency (TPR)")
        ax.set_title(f"Significant improvement characteristic {name}")
        # ax.plot([0,1],[0,1],color="gray",ls=":",label="Random")
    fname = f"{outdir}/SIC_sig_inj.png"
    ax.legend()
    fig.savefig(fname)

    
def plot_max_SIC(sig_percent, max_SIC, label="", outdir="./"):
    
    sig_percent = np.array(sig_percent)*100
    
    plt.figure(figsize=(7, 5))
    plt.plot(sig_percent, max_SIC, '-', label=label)  # Line color (default)
    plt.plot(sig_percent, max_SIC, 'x', color='black')  # Marker color (black)
    plt.xscale('log')
    plt.ylabel(r"max SIC")
    plt.xlabel("S/B (%)")
    plt.legend()
    plt.title(f"max SIC per signal significance")
    plt.savefig(f"{outdir}/maxSIC_sig_inj.png")
    

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
    

def plot_avg_max_SIC(sig_percent, max_SIC_list, label_list, outdir="./", title="Maximum significance improvement", tag=""):
    
    sig_percent = np.array(sig_percent)*100
    
    plt.figure(figsize=(7, 5))
    
    for i in range(len(max_SIC_list)):
    
        plt.plot(sig_percent, max_SIC_list[i], '-', label=label_list[i])  # Line color (default)
        plt.plot(sig_percent, max_SIC_list[i], 'x', color='black')  # Marker color (black)
    
    plt.xscale('log')
    plt.ylabel(r"max SIC")
    plt.xlabel("S/B (%)")
    plt.legend()
    plt.title(f"{title}")
    plt.savefig(f"{outdir}/avg_maxSIC{tag}.png")
    