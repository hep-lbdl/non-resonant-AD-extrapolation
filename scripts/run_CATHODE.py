import argparse
import numpy as np
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div_phys
from helpers.utils import load_nn_config
from semivisible_jet.utils import *
import torch
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    action="store",
    default=None,
    help=".npz file for input training samples and conditional inputs",
)
parser.add_argument(
    '-w', 
    "--weights",
    action="store",
    default="reweighting_run/weights.npz",
    help='Load weights.'
)
parser.add_argument(
    "-c",
    "--config",
    action="store",
    default=None,
    help="Classifier config file",
)
parser.add_argument(
    '-s', 
    "--samples",
    action="store",
    default=None,
    help='Directly load generated samples.'
)
parser.add_argument(
    '-m', 
    "--model",
    action="store_true",
    default=False,
    help='Load best trained MAF model.'
)
parser.add_argument(
    "--oversample",
    action="store_true",
    default=False,
    help="Oversampling",
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="CATHODE_run",
    help="output directory",
)
parser.add_argument(
    "--toy",
    action="store_true",
    default=False,
    help="Load toy samples.",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="Verbose enable DEBUG",
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

log_level = logging.DEBUG if args.verbose else logging.INFO
    
log = logging.getLogger("run")
log.setLevel(log_level)


def load_samples():
    # load input files
    inputs = np.load(args.input, mmap_mode='r')
    # data and MC
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    MC_context = inputs["MC_context"]
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    inputs.close()
    
    # Get feature and contexts from data
    data_feat_CR = data_feature[data_mask_CR]
    data_feat_SR = data_feature[data_mask_SR]
    data_cond_CR = data_context[data_mask_CR]
    data_cond_SR = data_context[data_mask_SR]
    # Get SR contexts from MC
    MC_cond_SR = MC_context[MC_mask_SR]

    if args.toy:
        data_feat_CR = data_feat_CR.reshape(-1,1)
        data_feat_SR = data_feat_SR.reshape(-1,1)
    
    return data_feat_CR, data_feat_SR, data_cond_CR, data_cond_SR, MC_cond_SR


def plot_reweighting(data_feat, pred_bkg, w_MC=None, RTag="SR"):

    feat = ["m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    for i in range(len(feat)):
        name = name_map()[feat[i]]
        if feat[i]=="m_jj":
            bins = np.linspace(0, 4000, 40)
        else:
            bins = np.linspace(0, 1, 20)
        
        # plot SR distribution
        plot_kwargs = {"bins":bins, "name": name, "tag":f"{feat[i]}_{RTag}_unweighted", "outdir":args.outdir}
        plot_kl_div_phys(data_feat[:,i], pred_bkg[:,i], f"true bkg {RTag}", f"predicted bkg {RTag}", **plot_kwargs)
        
        # plot reweigted SR distribution
        if w_MC is not None:
            plot_kwargs = {"w2":[w_MC], "bins":bins, "name": name, "tag":f"{feat[i]}_{RTag}_weighted", "outdir":args.outdir}
            plot_kl_div_phys(data_feat[:,i], pred_bkg[:,i], f"true bkg {RTag}", f"predicted bkg {RTag}", **plot_kwargs)


def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cpu")
    
    os.makedirs(args.outdir, exist_ok=True)
        
    # load input files
    data_feat_CR, data_feat_SR, data_cond_CR, data_cond_SR, MC_cond_SR = load_samples()

    # define useful variables
    nfeat = data_feat_CR.shape[1]
    ncond = data_cond_CR.shape[1]
    input_dim = nfeat+ncond if args.toy else nfeat
    n_sample = 1

    # Load existing model
    load_model = args.model

    # Load or sample predicted background
    if args.samples is None:
        
        if load_model:
            # MAF model path
            model_path = f"{args.outdir}/MAF_final_model.pt"
            # Check if a model exist
            if os.path.isfile(model_path):
                # Load the trained model
                logging.info("Loading a trained MAF...")
                MAF = torch.load(model_path)
                MAF.to(device)
            else:
                load_model = False

        if not load_model:   
            # Train a MAF for density estimation
            logging.info("Training a MAF to learn P(x|m)...")
            MAF = SimpleMAF(num_features=nfeat, num_context=ncond, device=device)
            MAF.train(data=data_feat_CR, cond=data_cond_CR, outdir=args.outdir, save_model=True)

        if args.verbose:
            # sample CR for from data
            pred_bkg_CR = MAF.sample(n_sample, data_cond_CR)
            # Plotting
            plot_reweighting(data_feat_CR, pred_bkg_CR, w_MC=None, RTag="CR")

        # sample from MAF
        n_sample = 1 if args.oversample else 1
        pred_bkg_SR = MAF.sample(n_sample, MC_cond_SR)
    
        # save generated samples
        np.savez(f"{args.outdir}/samples_data_feat_SR.npz", samples = pred_bkg_SR)
        log.debug(f"MAF generated {pred_bkg_SR.shape[0]} bkg events in the SR. Oversampling is not avaliable.")
        
    else:
        # Load samples
        pred_bkg_SR = np.load(args.samples)["samples"]
        log.info(f"{args.samples} loaded.")

    # load weights
    w_MC = np.load(args.weights)["weights"]
    
    # make reweighting plots if not using toy sampless
    if not args.toy:
        plot_reweighting(data_feat_SR, pred_bkg_SR, w_MC, RTag="SR")
    
    # Create training data set for the classifier.
    input_feat_x = np.concatenate([pred_bkg_SR, data_feat_SR], axis=0)
    if args.toy:
        input_cond_x = np.concatenate([MC_cond_SR, data_cond_SR], axis=0)
        input_x = np.concatenate([input_cond_x, input_feat_x], axis=1)
    else:
        input_x = input_feat_x

    # Create labels for the classifier.
    pred_bkg_SR_label = np.zeros(pred_bkg_SR.shape[0])
    data_feat_SR_label = np.ones(data_feat_SR.shape[0])
    input_y = np.hstack([pred_bkg_SR_label, data_feat_SR_label]).reshape(-1, 1)

    w_data = np.array([1.] * data_feat_SR.shape[0])
    input_weights = np.hstack([w_MC, w_data]).reshape(-1, 1)
    
    # Train a classifier for x.
    log.info("Training a classifer for signal vs background...")
    layers, lr, bs = load_nn_config(args.config)
    NN = Classifier(n_inputs=input_dim, layers=layers, learning_rate=lr, device=device, outdir=f"{args.outdir}/signal_significance")
    NN.train(input_x, input_y, weights=input_weights, batch_size=bs, save_model=True, model_name="0")

    log.info("CATHODE style extrapolation done!")
    
if __name__ == "__main__":
    main()