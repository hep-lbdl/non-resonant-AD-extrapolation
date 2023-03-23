import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div
import torch
import os

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
    data_CR = inputs["data_CR"]
    data_SR = inputs["data_SR"]
    cond_CR = inputs["cond_CR"]
    cond_SR = inputs["cond_SR"]
    inputs.close()
    
    # define useful variables
    num_features = data_CR.ndim
    num_cond_features = cond_CR.ndim
    num_samples = 1 # can set to higher values
    
    if args.samples is None:
        # train MAF
        MAF = SimpleMAF(num_features = 1, num_cond_features=2, device=device)
        MAF.train(data=data_CR, cond=cond_CR, plot=True, outdir=args.outdir)
        samples = MAF.sample(num_samples, cond_SR).reshape(len(cond_SR)) 
        # TODO: think about the reshape when oversample
        # save the best model

        # save generated samples
        np.savez(f"{args.outdir}/samples_gen.npz", samples = samples)
    else:
        # load samples
        samples = np.load(args.samples)["samples"].reshape(len(cond_SR))
    
    plot_kwargs = {"tag":"2DSR", "ymin":-15, "ymax":15, "outdir":args.outdir}
    plot_kl_div([data_SR], [samples], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
    
    # create labels for classifier
    samples_label = np.zeros(samples.shape)
    data_SR_label = np.ones(data_SR.shape)
    
    # create training data set for classifier
    input_bkg = np.stack([samples, samples_label], axis=-1)
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