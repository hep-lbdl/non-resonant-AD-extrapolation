import argparse
import numpy as np
from helpers.Classifier import Classifier
from helpers.plotting import plot_multi_dist
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
    default=None,
    help='Directly load generated weights.'
)
parser.add_argument(
    "-c",
    "--config",
    action="store",
    default=None,
    help="Classifier config file",
)
parser.add_argument(
    '-t', 
    "--trains",
    action="store",
    type=int,
    default=1,
    help='Number of trainings.'
)
parser.add_argument(
    "-e",
    "--evaluation",
    action="store_true",
    default=False,
    help="Only evaluate the reweighting classifier.",
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
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
    inputs = np.load(args.input)
    # data MC, and bkg
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    MC_feature = inputs["MC_feature"]
    MC_context = inputs["MC_context"]
    bkg_feature = inputs["bkg_feature"]
    bkg_context = inputs["bkg_context"]
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_CR = inputs["MC_mask_CR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    bkg_mask_SR = inputs["bkg_mask_SR"]
    # Signal injected
    sig_percent = inputs["sig_percent"]
    inputs.close()
    
    # Get feature and contexts from data
    data_feat_CR = data_feature[data_mask_CR]
    data_feat_SR = data_feature[data_mask_SR]
    data_cond_CR = data_context[data_mask_CR]
    data_cond_SR = data_context[data_mask_SR]

    # Get feature and contexts from MC
    MC_feat_CR = MC_feature[MC_mask_CR]
    MC_feat_SR = MC_feature[MC_mask_SR]
    MC_cond_CR = MC_context[MC_mask_CR]
    MC_cond_SR = MC_context[MC_mask_SR]
    
    # Get feature and contexts from MC
    bkg_feat_SR = bkg_feature[bkg_mask_SR]
    bkg_cond_SR = bkg_context[bkg_mask_SR]

    if args.toy:
        data_feat_CR = data_feat_CR.reshape(-1, 1)
        data_feat_SR = data_feat_SR.reshape(-1, 1)
        MC_feat_CR = MC_feat_CR.reshape(-1, 1)
        MC_feat_SR = MC_feat_SR.reshape(-1, 1)
        bkg_feat_SR = bkg_feat_SR.reshape(-1, 1)


    return data_feat_CR, data_feat_SR, data_cond_CR, data_cond_SR, MC_feat_CR, MC_feat_SR, MC_cond_CR, MC_cond_SR, bkg_feat_SR, bkg_cond_SR, sig_percent


def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    os.makedirs(args.outdir, exist_ok=True)
        
    data_feat_CR, data_feat_SR, data_cond_CR, data_cond_SR, MC_feat_CR, MC_feat_SR, MC_cond_CR, MC_cond_SR, bkg_feat_SR, bkg_cond_SR, sig_percent = load_samples()

    # define useful variables
    nfeat = data_feat_CR.shape[1]
    ncond = data_cond_CR.shape[1]
    input_dim = nfeat + ncond if args.toy else nfeat

    if args.weights is None:
        
        # create training data set for classifier
        input_feat_CR = np.concatenate([MC_feat_CR, data_feat_CR], axis=0)
        input_cond_CR = np.concatenate([MC_cond_CR, data_cond_CR], axis=0)
        input_x_train_CR = np.concatenate([input_cond_CR, input_feat_CR], axis=1)
        
        # create labels for classifier
        MC_CR_label = np.zeros(MC_cond_CR.shape[0]).reshape(-1,1)
        data_CR_label = np.ones(data_cond_CR.shape[0]).reshape(-1,1)
        
        input_y_train_CR = np.concatenate([MC_CR_label, data_CR_label], axis=0)
        
        if not args.evaluation:
            # train reweighting classifier
            log.info("Training a classifer for reweighting...")
            
            NN_reweight_train = Classifier(n_inputs=nfeat + ncond, layers=[64,64,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/reweighting")
            NN_reweight_train.train(input_x_train_CR, input_y_train_CR, save_model=True)
            
        # load the best trained model
        log.info("Loading the best classifer for reweighting...")
        
        model_path = f"{args.outdir}/reweighting/trained_AD_classifier.pt"
        NN_reweight = torch.load(f"{model_path}")
        NN_reweight.to(device)
        
        # evaluate classifier and calculate the weights
        input_x_test = np.concatenate([MC_cond_SR, MC_feat_SR], axis=1)
        
        w_SR = NN_reweight.evaluation(input_x_test)
        w_SR = (w_SR/(1.-w_SR)).flatten()
        
        # make validation plots in CR
        input_x_test_CR = np.concatenate([MC_cond_CR, MC_feat_CR], axis=1)

        w_CR = NN_reweight.evaluation(input_x_test_CR)
        w_CR = (w_CR/(1.-w_CR)).flatten()

        # save weights
        np.savez(f"{args.outdir}/SALAD_weights.npz", weights = w_SR)
        
        if not args.toy:
            
            os.makedirs(f"{args.outdir}/reweighting_plots", exist_ok=True)
            
            # only apply to the physics samples
            # plot reweigted distribution
            variables = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
            names = []
            units = []
            for var in variables:
                names.append(name_map()[var])
                units.append(unit_map()[var])
                
            ymaxs = [3000, 600, 6000] + [1]*4
            ymins = [800, 100] + [0]*5

            for i in [0,1]:

                hlist = [MC_cond_SR[:,i], MC_cond_SR[:,i], bkg_cond_SR[:,i], MC_cond_CR[:,i], MC_cond_CR[:,i], data_cond_CR[:,i]]
                weights = [None, w_SR, None, None, w_CR, None]
                labels = [f"MC SR", "Reweighted MC SR", "True bkg SR", "MC CR", "Reweighted MC CR", "Data CR"]
                htype = ["step", "stepfilled", "step"]*2
                lstyle = ["-"]*3 + ["--"]*3
                plot_kwargs = {"title":f"Reweighted MC vs data for {names[i]}, S/B={sig_percent*1e2:.1f}%", "name":f"MC vs data reweighting {names[i]}", "xlabel":f"{names[i]} {units[i]}", "ymin":ymins[i], "ymax":ymaxs[i], "outdir":f"{args.outdir}/reweighting_plots"}
                plot_multi_dist(hlist, labels, weights=weights, htype=htype, lstyle=lstyle, **plot_kwargs)

            for i in range(nfeat):

                hlist = [MC_feat_SR[:,i], MC_feat_SR[:,i], bkg_feat_SR[:,i], MC_feat_CR[:,i], MC_feat_CR[:,i], data_feat_CR[:,i]]
                weights = [None, w_SR, None, None, w_CR, None]
                labels = [f"MC SR (Num Events: {len(MC_feat_SR[:,i]):.1e})", "reweighted MC SR", "true bkg SR", "MC CR", "reweighted MC CR", "data CR"]
                htype = ["step", "stepfilled", "step"]*2
                lstyle = ["-"]*3 + ["--"]*3
                plot_kwargs = {"title":f"Reweighted MC vs data for {names[i+2]}, S/B={sig_percent*1e2:.3f}%", "name":f"MC vs data reweighting {variables[i+2]}", "xlabel":f"{names[i+2]} {units[i+2]}", "ymin":0, "ymax":ymaxs[i+2], "outdir":f"{args.outdir}/reweighting_plots"}
                plot_multi_dist(hlist, labels, weights=weights, htype=htype, lstyle=lstyle, **plot_kwargs)


    else:
        # load weights
        log.info("Loading SALAD weights...")
        w_SR = np.load(args.weights)["weights"]

    
    log.info("Training a classifer for signal vs background...")
    
    # create training data set for classifier
    input_feat_SR = np.concatenate([MC_feat_SR, data_feat_SR], axis=0)
    if args.toy:
        input_cond_SR = np.concatenate([MC_cond_SR, data_cond_SR], axis=0)
        input_x_train_SR = np.concatenate([input_cond_SR, input_feat_SR], axis=1)
    else:
        input_x_train_SR = input_feat_SR
    
    # create labels for classifier
    MC_SR_label = np.zeros(MC_cond_SR.shape[0]).reshape(-1,1)
    data_SR_label = np.ones(data_cond_SR.shape[0]).reshape(-1,1)
    input_y_train_SR = np.concatenate([MC_SR_label, data_SR_label], axis=0)

    w_data = np.array([1.]*data_feat_SR.shape[0])
    input_weights = np.hstack([w_SR, w_data]).reshape(-1, 1)
    
    # Train the AD Classifier
    log.info(f"Training a classifer for signal vs background...")
    log.info(f"Ensamble size: {args.trains}")
    layers, lr, bs = load_nn_config(args.config)
    
    for i in range(args.trains):
        # Train a classifier for x, m1 and m2.
        NN = Classifier(n_inputs=input_dim, layers=layers, learning_rate=lr, device=device, outdir=f"{args.outdir}/signal_significance")
        NN.train(input_x_train_SR, input_y_train_SR, weights=input_weights, batch_size=bs, save_model=True, model_name=f"{i}")

    log.info("SALAD extrapolation done!")
    
if __name__ == "__main__":
    main()