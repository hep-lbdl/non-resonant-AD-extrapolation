import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kl_div
import os

os.makedirs(os.path.dirname("./plots"), exist_ok=True)

def get_kl_div(p,q):
    div_arr = np.where(np.logical_and(p>0,q>0),kl_div(p,q),0)
    return np.sum(div_arr)

def plot_gen_full_bkg(samples, x1, x2):
    sample2 = samples[:,0]
    y1 = sample2[:,0]
    y2 = sample2[:,1]
    plt.figure(figsize=(6,6))
    plt.scatter(x1, x2, alpha = 0.2, label = 'true bkg')
    plt.scatter(y1, y2, alpha = 0.2, label = 'generated bkg')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.axvline(x=1, color='r', linestyle='-')
    plt.legend()
    plt.title("Generated full bkg distribution from training full bkg samples.")
    plt.show()
    plt.savefig('plots/gen_full_bkg.png')
    plt.close
    
def plot_gen_SRfromCR_bkg(samples, x1, x2):
    sample2 = samples[:,0]
    y1 = sample2[:,0]
    y2 = sample2[:,1]
    plt.figure(figsize=(6,6))
    plt.scatter(x1, x2, alpha = 0.2, label = 'true bkg')
    plt.scatter(y1, y2, alpha = 0.2, label = 'generated bkg')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.axvline(x=1, color='r', linestyle='-')
    plt.legend()
    plt.title("Generated full bkg distribution from training CR samples.")
    plt.show()
    plt.savefig('plots/gen_SRfromCR_bkg.png')
    plt.close
    
def plot_gen_SR_bkg_in_y_random(samples, Y_SR):
    sample2 = np.array(samples[:,0])
    plt.figure(figsize=(6,4))
    bins = np.linspace(-5, 5, 50)
    plt.hist(sample2, bins = bins, density = True, histtype='step', label='generated SR')
    plt.hist(Y_SR, bins = bins, density = True, histtype='step', label='true SR')
    plt.title("Generated bkg distribution in y = random.")
    plt.legend()
    plt.show
    plt.savefig('plots/gen_SR_bkg_in_y_random.png')
    plt.close
    
def plot_gen_SR_bkg_in_y_cond(samples, Y_SR, k, q):
    sample2 = np.array(samples[:,0])
    bins = np.linspace(-5, 5, 50)
    plt.figure(figsize=(6,4))
    plt.hist(sample2, bins = bins, density = True, histtype='step', label='generated SR')
    plt.hist(Y_SR, bins = bins, density = True, histtype='step', label='true SR')
    plt.title(f"Generated bkg in y = N(${k}\\alpha$+{q}$\\beta$, 1)")
    plt.legend()
    plt.show
    plt.savefig(f'plots/gen_SR_bkg_in_y_cond{k*10}.png')
    plt.close
    
def plot_gen_SR_bkg_in_y_cond_list(samples_list, Y_SR, k, q):
    colors = ['blue', 'green', 'slategrey', 'steelblue']
    sample2 = [] 
    for samples in samples_list:
        sample2.append(np.array(samples[:,0]))
    N = len(sample2)
    if N==len(Y_SR) and N==len(k) and N==len(q) and N<len(colors):
        bins = np.linspace(-6, 8, 50)
        fig, ax1 = plt.subplots(figsize=(12,8))
        for i in range(N):
            c0, cbins, _ = ax1.hist(Y_SR[i], bins = bins, density = True, histtype='stepfilled', alpha = 0.5, color=f"light{colors[i]}", label=f"true SR, k={k[i]} q={q[i]}")
            c1, cbins, _ = ax1.hist(sample2[i], bins = bins, density = True, histtype='step', color=colors[i], label=f"generated SR, k={k[i]} q={q[i]}")
            kl_div = get_kl_div(c0,c1)
            ax1.hist(sample2[i], bins = bins, density = True, histtype='step', color=colors[i], label=f"kl div={kl_div:.3f}")
        ax1.set_title("Generated bkg in y = Normal(mean = $k\\alpha$+q$\\beta$, $\\sigma$ = 1)", fontsize = 14)
        ax1.set_xlabel("y")
        plt.legend(loc='upper left', fontsize = 14)
        plt.show
        plt.savefig('plots/gen_SR_bkg_in_y_cond.png')
        plt.close
    else:
        print("Wrong input lists!")