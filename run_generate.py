import argparse
import numpy as np
from helpers.SimpleMAF import SimpleMAF
import torch
import os
import logging
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    help="home folder for input training samples and conditional inputs",
    default="/global/cfs/cdirs/m3246/rmastand/bkg_extrap/redo/data/"
)

parser.add_argument(
    "-s",
    "--signal",
    default=None,
    help="signal fraction",
)

parser.add_argument(
    "-c",
    "--config",
    help="Generate flow config file",
    default="configs/generate_physics.yml"
)

parser.add_argument(
    '-l', 
    "--load_model",
    default=False,
    help='Load best trained model.'
)

parser.add_argument(
    '-m', 
    "--model_path",
    help='Path to best trained model'
)

parser.add_argument(
    "--oversample",
    default=1,
    help="Oversampling",
)
parser.add_argument(
    "-o",
    "--outdir",
    help="output directory",
    default="/global/cfs/cdirs/m3246/rmastand/bkg_extrap/redo/"
)

parser.add_argument(
    "-v",
    "--verbose",
    default=False,
    help="Verbose enable DEBUG",
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

log_level = logging.DEBUG if args.verbose else logging.INFO
    
log = logging.getLogger("run")
log.setLevel(log_level)

def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    model_dir = f"{args.outdir}/models/"
    samples_dir = f"{args.outdir}/samples/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
        
    # load input files
    data_events = np.load(f"{args.input}/data_{args.signal}.npz")
    data_events_cr = data_events["data_events_cr"]
    data_events_sr = data_events["data_events_sr"]
    
    mc_events = np.load(f"{args.input}/mc_events.npz")
    mc_events_sr = mc_events["mc_events_sr"]
    
    print("Working with s/b =", args.signal, ". CR has", len(data_events_cr), "events, SR has", len(data_events_sr), "events.")

    # Train flow in the CR
    # To do the closure tests, we need to withhold a small amount of CR data
    n_withold = 10000 
    n_context = 2
    n_features = 5
    
    data_context_cr_train = data_events_cr[:-n_withold,:n_context]
    data_context_cr_test = data_events_cr[-n_withold:,:n_context]
    data_feature_cr_train = data_events_cr[:-n_withold,n_context:]
    data_feature_cr_test = data_events_cr[-n_withold:,n_context:]
    
    mc_context_sr = mc_events_sr[:,:n_context]
    mc_feature_sr = mc_events_sr[:,n_context:]
    
    if args.config is not None:
        with open(args.config, 'r') as stream:
            params = yaml.safe_load(stream)
            n_layers = params["n_layers"]
            n_hidden_features = params["n_hidden_features"]
            learning_rate = params["learning_rate"]
            batch_size = params["batch_size"]
            n_epochs = params["n_epochs"]
    else:
        n_layers = 1
        n_hidden_features = 8
        learning_rate = 0.0001
        batch_size = 256
        n_epochs = 20
         
    # Define the flow
    MAF = SimpleMAF(num_features=n_features, num_context=n_context, device=device, num_layers=n_layers, num_hidden_features=n_hidden_features, learning_rate = learning_rate)
    
    # Model in
    load_model = args.load_model
    
    if load_model:
        # Check if a model exist
        if os.path.isfile(args.model_path):
            # Load the trained model
            print("Loading in model...")
            MAF = torch.load(args.model_path)
            MAF.to(device)
        else:
            load_model = False

    if not load_model:   
        print("Training Generate model...")

        MAF.train(data=data_feature_cr_train, cond=data_context_cr_train, batch_size=batch_size, n_epochs=n_epochs, outdir=model_dir, save_model=True, model_name="generate_best")
        print("Done training!")
        
    print("Making samples...")
    # sample CR from data
    pred_bkg_CR = MAF.sample(1, data_context_cr_test)
    np.savez(f"{samples_dir}/generate_CR_closure_s{args.signal}.npz", target_cr=data_feature_cr_test, generate_cr=pred_bkg_CR)

    # sample from MAF
    n_sample = 1 if args.oversample else 1
    pred_bkg_SR = MAF.sample(n_sample, mc_context_sr)

    # save generated samples
    np.savez(f"{samples_dir}/generate_SR_s{args.signal}.npz", samples = pred_bkg_SR)
    
    print("All done.")

if __name__ == "__main__":
    main()