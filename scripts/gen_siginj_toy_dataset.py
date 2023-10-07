import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import *
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
    help="output directory",
)
parser.add_argument(
    '-t', 
    "--test",
    action="store_true",
    default=False,
    help='Generate test datasets.'
)
parser.add_argument(
    '-s', 
    "--supervised",
    action="store_true",
    default=False,
    help='Generate supervised datasets.'
)
args = parser.parse_args()



def X(k, theta, x1, x2, n, sigma=1):
    return np.random.normal(k*(cos(theta)*x1 + sin(theta)*x2), sigma, n).astype(dtype=np.float32)


def main():
    # Make a toy data set with k=0.5 and theta=pi/4
    
    bkg_mean = -1
    bkg_std = 2
    MC_mean = -1
    MC_std = 2.8
    sig_mean = 2.8
    sig_std = 0.5

    os.makedirs(args.outdir, exist_ok=True)
    
    if args.test:
        N1 = 500000
    else:
        N1 = 100000
        
        # MC
        MC_m1 = np.random.normal(MC_mean, MC_std, N1).astype(dtype=np.float32)
        MC_m2 = np.random.normal(MC_mean, MC_std, N1).astype(dtype=np.float32)
        MC_context = np.stack([MC_m1, MC_m2], axis = -1)

        MC_mask_CR = np.logical_not((MC_m1 > 1) & (MC_m2 > 1))
        MC_mask_SR = ((MC_m1 > 1) & (MC_m2 > 1))

        MC_feature = X(0.5, pi/4, MC_m1, MC_m2, N1)
    

    # data bkg-only
    bkg_m1 = np.random.normal(bkg_mean, bkg_std, N1).astype(dtype=np.float32)
    bkg_m2 = np.random.normal(bkg_mean, bkg_std, N1).astype(dtype=np.float32)
    
    bkg_feature = X(0.5, pi/4, bkg_m1, bkg_m2, n=N1)
    bkg_context = np.stack([bkg_m1, bkg_m2], axis = -1)
    bkg_mask_CR = np.logical_not((bkg_m1 > 1) & (bkg_m2 > 1))
    bkg_mask_SR = ((bkg_m1 > 1) & (bkg_m2 > 1))
    

    # initialize lists
    if args.test:
        sig_percent_list = [1]
    elif args.supervised:
        sig_percent_list = [1]*20
    else:
        sig_percent_list = np.logspace(np.log10(0.0005),np.log10(0.01),10).round(5)
    
    data_feature_list = []
    MC_feature_list = []
    sig_feature_list = []
    bkg_feature_list = []
    data_cond_m1_list = []
    data_cond_m2_list = []
    MC_cond_m1_list = []
    MC_cond_m2_list = []
    sig_cond_m1_list = []
    sig_cond_m2_list = []
    bkg_cond_m1_list = []
    bkg_cond_m2_list = []
    labels = []
    
    num=0
    for s in sig_percent_list:
        
        # Total number of signal
        N2 = int(s * N1)
        
        if N2>0:

            sig_m1 = np.random.normal(sig_mean, sig_std, N2).astype(dtype=np.float32)
            sig_m2 = np.random.normal(sig_mean, sig_std, N2).astype(dtype=np.float32)

            data_m1 = np.hstack([bkg_m1, sig_m1])
            data_m2 = np.hstack([bkg_m2, sig_m2])

            sig_feature = X(0.5, pi/4, sig_m1, sig_m2, n=N2)
            sig_context = np.stack([sig_m1, sig_m2], axis = -1)
            sig_mask_CR = np.logical_not((sig_m1 > 1) & (sig_m2 > 1))
            sig_mask_SR = ((sig_m1 > 1) & (sig_m2 > 1))
            
        else:
            
            data_m1 = bkg_m1
            data_m2 = bkg_m2
            
            sig_feature = np.empty((0))
            sig_context = np.empty((0))
            sig_mask_CR = np.empty((0))
            sig_mask_SR = np.empty((0))
        
        data_feature = X(0.5, pi/4, data_m1, data_m2, n=N1+N2)
        data_context = np.stack([data_m1, data_m2], axis = -1)
        data_mask_CR = np.logical_not((data_m1 > 1) & (data_m2 > 1))
        data_mask_SR = ((data_m1 > 1) & (data_m2 > 1))


        print(f"S/B={s}, N1+N2={N1+N2}, data_context: {data_context.shape}, data_feature: {data_feature.shape}")
        
        if args.test:
            np.savez(f"./{args.outdir}/test_inputs.npz", bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, bkg_mask_SR=bkg_mask_SR, sig_mask_SR = sig_mask_SR, sig_percent=s)
        
        if args.supervised:
            np.savez(f"./{args.outdir}/supervised_inputs_{num}.npz", bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, bkg_mask_SR=bkg_mask_SR, sig_mask_SR = sig_mask_SR, sig_percent=s)
            
        else:
            np.savez(f"./{args.outdir}/inputs_s{num}.npz", data_feature=data_feature, data_context=data_context, MC_feature=MC_feature, MC_context=MC_context, bkg_feature=bkg_feature, bkg_context=bkg_context, sig_feature = sig_feature, sig_context = sig_context, data_mask_CR=data_mask_CR, data_mask_SR=data_mask_SR, MC_mask_CR=MC_mask_CR, MC_mask_SR=MC_mask_SR, bkg_mask_CR=bkg_mask_CR, bkg_mask_SR=bkg_mask_SR, sig_mask_CR = sig_mask_CR, sig_mask_SR = sig_mask_SR, sig_percent=s)
        
        
            data_feature_list.append(data_feature)
            MC_feature_list.append(MC_feature)
            data_cond_m1_list.append(data_m1)
            data_cond_m2_list.append(data_m2)
            MC_cond_m1_list.append(MC_m1)
            MC_cond_m2_list.append(MC_m2)
            
            sig_feature_list.append(sig_feature)          
            sig_cond_m1_list.append(sig_m1)
            sig_cond_m2_list.append(sig_m2)
        
        labels.append(f"percent signal={s}")
        
        num = num +1

        
    if args.test is False and args.supervised is False:
        # Plot data and MC contexts
        plot_kwargs = {"name":f"data_vs_mc_m1_s{num}", "title":f"Input context: data $m_1 = N({bkg_mean},{bkg_std})$ and MC $m_1 = N({MC_mean},{MC_std}) with S/B = {s}$", "xlabel":r"$m_1$", "ymin":-15, "ymax":15, "outdir":args.outdir}
        plot_multi_data_MC_dist(data_cond_m1_list, MC_cond_m1_list, labels, **plot_kwargs)

        plot_kwargs = {"name":f"data_vs_mc_m2_s{num}", "title":f"Input context: data $m_2 = N({bkg_mean},{bkg_std})$ and MC $m_2 = N({MC_mean},{MC_std})$ with S/B = {s}", "xlabel":r"$m_2$", "ymin":-15, "ymax":15, "outdir":args.outdir}
        plot_multi_data_MC_dist(data_cond_m2_list, MC_cond_m2_list, labels, **plot_kwargs)

        # Plot all data and MC features
        plot_kwargs = {"name":"data_vs_mc_x", "title":f"Input feature $x = N(k(m_1, m_2), 1)$", "xlabel":"x", "ymin":-15, "ymax":15, "outdir":args.outdir}
        plot_multi_data_MC_dist(data_feature_list, MC_feature_list, labels, **plot_kwargs)

        # Plot sig and bkg contexts
        plot_kwargs = {"name":"sig_vs_bkg_m1", "title":f"Input context: sig $m_1 = N({sig_mean},{sig_std})$ and bkg $m_1 = N({bkg_mean},{bkg_std})$", "xlabel":"m1", "ymin":-10, "ymax":10, "outdir":args.outdir}
        plot_sig_bkg_dist(sig_cond_m1_list, bkg_m1, labels, **plot_kwargs)        
        
        # Plot all data and MC features
        plot_kwargs = {"name":"sig_vs_bkg_x", "title":f"Input feature $x = N(k(m_1, m_2), 1)$", "xlabel":"x", "ymin":-10, "ymax":10, "outdir":args.outdir}
        plot_sig_bkg_dist(sig_feature_list, bkg_feature, labels, **plot_kwargs)

if __name__ == "__main__":
    main()