import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import plot_kl_div, plot_multi_data_MC_dist
import os

# Total number of events
N1 = 100000
# Total number of signal
N2 = 1000

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

    # data
    m1 = np.random.normal(0, 1, N1).astype(dtype=np.float32)
    m2 = np.random.normal(0, 1, N1).astype(dtype=np.float32)
    data_context = np.stack([m1, m2], axis = -1)

    mask_CR1D = m1 < 1
    mask_SR1D = m1 > 1

    data_mask_CR = np.logical_not((m1 > 1) & (m2 > 1))
    data_mask_SR = ((m1 > 1) & (m2 > 1))

    # MC
    MC_m1 = np.random.normal(0, 1.8, N1).astype(dtype=np.float32)
    MC_m2 = np.random.normal(0, 1.8, N1).astype(dtype=np.float32)
    MC_context = np.stack([MC_m1, MC_m2], axis = -1)

    MC_mask_CR = np.logical_not((MC_m1 > 1) & (MC_m2 > 1))
    MC_mask_SR = ((MC_m1 > 1) & (MC_m2 > 1))
    
    k_list = [0, 0.2, 0.5, 0.8, 1, 1.5, 2]
    data_feature_list = []
    MC_feature_list = []
    labels = []
    
    num=0
    for k in k_list:
    
        data_feature = X(k, pi/4, m1, m2)
        MC_feature = X(k, pi/4, MC_m1, MC_m2)
        
        data_feature_list.append(data_feature)
        MC_feature_list.append(MC_feature)
        labels.append(f"k={k}")

        np.savez(f"./{args.outdir}/inputs_k{num}.npz", data_feature=data_feature, data_context=data_context, MC_feature=MC_feature, MC_context=MC_context, data_mask_CR=data_mask_CR, data_mask_SR=data_mask_SR, MC_mask_CR=MC_mask_CR, MC_mask_SR=MC_mask_SR)
        
        num = num +1

    # Plot data and MC contexts
    plot_kwargs = {"name":"data_vs_mc_m1", "title":r"Input context: data $m_1 = N(0,1)$ and MC $m_1 = N(0,1.8)$", "xlabel":r"$m_1$", "ymin":-15, "ymax":15, "outdir":args.outdir}
    plot_multi_data_MC_dist([m1], [MC_m1], [""], **plot_kwargs)

    plot_kwargs = {"name":"data_vs_mc_m2", "title":r"Input context: data $m_2 = N(0,1)$ and MC $m_2 = N(0,1.8)$", "xlabel":r"$m_2$", "ymin":-15, "ymax":15, "outdir":args.outdir}
    plot_multi_data_MC_dist([m2], [MC_m2], [""], **plot_kwargs)
    
    # Plot all data and MC features
    plot_kwargs = {"name":"data_vs_mc_x", "title":f"Input feature x with k={k}", "xlabel":"x", "ymin":-15, "ymax":15, "outdir":args.outdir}
    plot_multi_data_MC_dist(data_feature_list, MC_feature_list, labels, **plot_kwargs)

if __name__ == "__main__":
    main()