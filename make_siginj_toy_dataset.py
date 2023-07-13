import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import plot_kl_div, plot_multi_data_MC_dist
import os

# Total number of events
N1 = 500000

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
    help="output directory",
)
args = parser.parse_args()



def X(k, theta, x1, x2, n=N1, sigma=1):
    return np.random.normal(k*(cos(theta)*x1 + sin(theta)*x2), sigma, n).astype(dtype=np.float32)


def main():
    # Make a toy data set with k=0.5 and theta=pi/4

    os.makedirs(args.outdir, exist_ok=True)
    
    # data bkg-only
    m1 = np.random.normal(0, 1, N1).astype(dtype=np.float32)
    m2 = np.random.normal(0, 1, N1).astype(dtype=np.float32)
    
    bkg_feature = X(0.5, pi/4, m1, m2, n=N1)
    bkg_context = np.stack([m1, m2], axis = -1)
    bkg_mask_CR = np.logical_not((m1 > 1) & (m2 > 1))
    bkg_mask_SR = ((m1 > 1) & (m2 > 1))
    
    # MC
    MC_m1 = np.random.normal(0, 1.8, N1).astype(dtype=np.float32)
    MC_m2 = np.random.normal(0, 1.8, N1).astype(dtype=np.float32)
    MC_context = np.stack([MC_m1, MC_m2], axis = -1)

    MC_mask_CR = np.logical_not((MC_m1 > 1) & (MC_m2 > 1))
    MC_mask_SR = ((MC_m1 > 1) & (MC_m2 > 1))
    
    MC_feature = X(0.5, pi/4, MC_m1, MC_m2)
    

    # initialize lists
    sig_percent_list = [0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050]
    data_feature_list = []
    MC_feature_list = []
    data_cond_m1_list = []
    data_cond_m2_list = []
    MC_cond_m1_list = []
    MC_cond_m2_list = []
    labels = []
    
    num=0
    for s in sig_percent_list:
        
        # Total number of signal
        N2 = int(s * N1)
        
        if N2>0:

            sig_m1 = np.random.normal(3, 0.5, N2).astype(dtype=np.float32)
            sig_m2 = np.random.normal(3, 0.5, N2).astype(dtype=np.float32)

            data_m1 = np.hstack([m1, sig_m1])
            data_m2 = np.hstack([m2, sig_m2])

            sig_feature = X(0.5, pi/4, sig_m1, sig_m2, n=N2)
            sig_context = np.stack([sig_m1, sig_m2], axis = -1)
            sig_mask_CR = np.logical_not((sig_m1 > 1) & (sig_m2 > 1))
            sig_mask_SR = ((sig_m1 > 1) & (sig_m2 > 1))
            
        else:
            
            data_m1 = m1
            data_m2 = m2
            
            sig_feature = np.empty((0))
            sig_context = np.empty((0))
            sig_mask_CR = np.empty((0))
            sig_mask_SR = np.empty((0))
        
        data_feature = X(0.5, pi/4, data_m1, data_m2, n=N1+N2)
        data_context = np.stack([data_m1, data_m2], axis = -1)
        data_mask_CR = np.logical_not((data_m1 > 1) & (data_m2 > 1))
        data_mask_SR = ((data_m1 > 1) & (data_m2 > 1))


        print(f"s={s}, N1+N2={N1+N2}, m1: {data_m1.shape}, m2: {data_m2.shape}, data_context: {data_context.shape}")
        
        np.savez(f"./{args.outdir}/inputs_s{num}.npz", data_feature=data_feature, data_context=data_context, MC_feature=MC_feature, MC_context=MC_context, bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, data_mask_CR=data_mask_CR, data_mask_SR=data_mask_SR, MC_mask_CR=MC_mask_CR, MC_mask_SR=MC_mask_SR, bkg_mask_CR=bkg_mask_CR, bkg_mask_SR=bkg_mask_SR, sig_mask_CR = sig_mask_CR, sig_mask_SR = sig_mask_SR)
        
        data_feature_list.append(data_feature)
        MC_feature_list.append(MC_feature)
        data_cond_m1_list.append(data_m1)
        data_cond_m2_list.append(data_m2)
        MC_cond_m1_list.append(MC_m1)
        MC_cond_m2_list.append(MC_m2)
        
        labels.append(f"percent signal={s}")
        
        num = num +1

    # Plot data and MC contexts
    plot_kwargs = {"name":f"data_vs_mc_m1_s{num}", "title":f"Input context: data $m_1 = N(0,1)$ and MC $m_1 = N(0,1.8) with S/B = {s}$", "xlabel":r"$m_1$", "ymin":-15, "ymax":15, "outdir":args.outdir}
    plot_multi_data_MC_dist(data_cond_m1_list, MC_cond_m1_list, labels, **plot_kwargs)

    plot_kwargs = {"name":f"data_vs_mc_m2_s{num}", "title":f"Input context: data $m_2 = N(0,1)$ and MC $m_2 = N(0,1.8)$ with S/B = {s}", "xlabel":r"$m_2$", "ymin":-15, "ymax":15, "outdir":args.outdir}
    plot_multi_data_MC_dist(data_cond_m2_list, MC_cond_m2_list, labels, **plot_kwargs)

    # Plot all data and MC features
    plot_kwargs = {"name":"data_vs_mc_x", "title":f"Input feature $x = N(k(m_1, m_2), 1)$", "xlabel":"x", "ymin":-15, "ymax":15, "outdir":args.outdir}
    plot_multi_data_MC_dist(data_feature_list, MC_feature_list, labels, **plot_kwargs)

if __name__ == "__main__":
    main()