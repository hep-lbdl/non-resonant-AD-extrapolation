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
    plt.title("Generated bkg distribution in y = random.")
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
    plt.title(f"Generated bkg in y = N(${k}\\alpha$+{q}$\\beta$, 1)")
    plt.legend()
    plt.show
    plt.savefig(f'plots/gen_SR_bkg_in_y_cond{k*10}.pdf')
    plt.close

def pi_to_string(theta):
    return f"({str(Fraction(theta/pi))})$\\pi$"
    
def plot_kl_div(Y_list, Y_list2, Y_label, Y_label2, k, theta=None, title="Normal($k(cos\\theta\\alpha + sin\\theta\\beta)$, 1)", tag = "", ymin=-6, ymax=10, outdir="plots", *args, **kwargs):
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
            
            c0, cbins, _ = ax1.hist(Y_list[i], bins = bins, density = True, histtype='step', color=colors[i], label=f"{Y_label}, {label_k}")
            c1, cbins, _ = ax1.hist(Y_list2[i], bins = bins, density = True, histtype='stepfilled', alpha = 0.3, color=colors[i], label=f"{Y_label2}, {label_k}")
            kl_div = get_kl_div(c0,c1)
            ax1.hist(Y_list2[i], bins = bins, density = True, histtype='stepfilled', alpha = 0, color=colors[i], label=f"kl div={kl_div:.3f}")
        ax1.set_title(f"Background in y = {title}", fontsize = 14)
        ax1.set_xlabel("y")
        plt.legend(loc='upper left', fontsize = 9)
        plt.show
        plot_name = f"{outdir}/{Y_label}_{Y_label2}_{tag}.pdf"
        plt.savefig(plot_name.replace(" ", "_"))
        plt.close()
    else:
        print("Wrong input lists!")
        
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