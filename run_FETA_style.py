import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div
import torch
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    action="store",
    help=".npz file for input training samples and conditional inputs",
)
parser.add_argument(
    '-s', 
    "--samples",
    action="store",
    default=None,
    help='Directly load generated samples.'
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
    help="output directory",
)
args = parser.parse_args()


def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    # load input files
    inputs = np.load(args.input)
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    MC_feature = inputs["MC_feature"]
    MC_context = inputs["MC_context"]
    mask_CR = inputs["mask_CR"]
    mask_SR = inputs["mask_SR"]
    inputs.close()
    
    data_CR = data_feature[mask_CR]
    data_SR = data_feature[mask_SR]
    data_cond_CR = data_context[mask_CR]
    data_cond_SR = data_context[mask_SR]
    MC_CR = MC_feature[mask_CR]
    MC_SR = MC_feature[mask_SR]
    MC_cond_CR = MC_context[mask_CR]
    MC_cond_SR = MC_context[mask_SR]
    
    # define useful variables
    nfeat = data_CR.ndim
    ncond = data_cond_CR.ndim
    num_samples = 1 # can set to higher values
    
    if args.samples is None:
        # train base density flow
        base_density_flow = SimpleMAF(num_features = nfeat, num_context=ncond, device=device)
        base_density_flow.train(data=MC_feature, cond = MC_context, plot=True, outdir=args.outdir)
        
        # train trasport flow
        transport_flow = SimpleMAF(num_features = nfeat, num_context=ncond, base_dist=base_density_flow.flow, device=device)
        transport_flow.train(data=data_CR, cond=data_cond_CR, plot=True, outdir=args.outdir)
        
        # sample from trasport flow
        transport_data_SR, _ = transport_flow.flow._transform.inverse(torch.tensor(MC_SR.reshape(-1, 1)).to(device), torch.tensor(MC_cond_SR).to(device))
        pred_bkg_SR = transport_data_SR.detach().cpu().numpy().flatten()
        
        # save generated samples
        np.savez(f"{args.outdir}/samples_data_SR.npz", samples = pred_bkg_SR)
    else:
        # load samples
        pred_bkg_SR = np.load(args.samples)["samples"]

    print(f"shape of pred_bkg_SR: {pred_bkg_SR.shape}")
    print(f"shape of data_SR: {data_SR.shape}")
    
    plot_kwargs = {"tag":"2DSR", "ymin":-15, "ymax":15, "outdir":args.outdir}
    plot_kl_div([data_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
    
    # create labels for classifier
    pred_bkg_SR_label = np.zeros(pred_bkg_SR.shape)
    data_SR_label = np.ones(data_SR.shape)
    
    # create training data set for classifier
    input_bkg = np.stack([pred_bkg_SR, pred_bkg_SR_label], axis=-1)
    input_data = np.stack([data_SR, data_SR_label], axis=-1)
    
    input_x = np.vstack([input_bkg, input_data])
    
    # train classifier
    NN = Classifier(n_inputs=1, device=device)
    NN.train(input_x, plot=True, outdir=args.outdir)
    
    # evaluate classifier
    # TODO: properly generate test dataset
    x_test = input_x[:, 0].reshape(-1, 1)
    y_test = input_x[:, -1].reshape(-1, 1)
    NN.evaluation(x_test, y_test, outdir=args.outdir, plot=True)

if __name__ == "__main__":
    main()