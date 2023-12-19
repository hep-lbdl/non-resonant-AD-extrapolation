import argparse
import numpy as np
from helpers.SimpleMAF import SimpleMAF
import torch
import os
import logging
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-i","--indir",help="working folder",
    default="/global/cfs/cdirs/m3246/rmastand/bkg_extrap/redo/"
)
parser.add_argument("-s","--signal",default=None,help="signal fraction",)
parser.add_argument("-c","--config",help="Generate flow config file",default="configs/generate_physics.yml")
parser.add_argument('-l', "--load_model",action='store_true',help='Load best trained model.')
parser.add_argument('-m',  "--model_path",help='Path to best trained model')
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument("-o","--oversample",help="How much to oversample the model",default=1)
parser.add_argument( "-v", "--verbose",default=False,help="Verbose enable DEBUG")
parser.add_argument("-cu", "--cuda_slot", help = "cuda_slot")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_slot)

logging.basicConfig(level=logging.INFO)
log_level = logging.DEBUG if args.verbose else logging.INFO
log = logging.getLogger("run")
log.setLevel(log_level)

def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    static_data_dir = f"{args.indir}/data/"
    seeded_data_dir = f"{args.indir}/data/seed{args.gen_seed}/"
    model_dir = f"{args.indir}/models/seed{args.gen_seed}/"
    samples_dir = f"{args.indir}/samples/seed{args.gen_seed}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
        
    # load input files
    data_events = np.load(f"{seeded_data_dir}/data_{args.signal}.npz")
    data_events_cr = data_events["data_events_cr"]
    
    mc_events = np.load(f"{static_data_dir}/mc_events.npz")
    mc_events_sr = mc_events["mc_events_sr"]
    
    print("Working with s/b =", args.signal, ". CR has", len(data_events_cr), "events.")

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
    
    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)
         
    # Define the flow
    MAF = SimpleMAF(num_features=n_features, num_context=n_context, device=device, num_layers=params["n_layers"], num_hidden_features=params["n_hidden_features"], learning_rate = params["learning_rate"])
    
    # Model in
    load_model = args.load_model
    
    if load_model:
        # Check if a model exist
        if os.path.isfile(args.model_path):
            # Load the trained model
            print(f"Loading in model from {args.model_path}...")
            MAF = torch.load(args.model_path)
            MAF.to(device)
        else:
            load_model = False

    if not load_model:   
        print("Training Generate model...")

        MAF.train(data=data_feature_cr_train, cond=data_context_cr_train, batch_size=params["batch_size"], n_epochs=params["n_epochs"], outdir=model_dir, save_model=True, model_name=f"generate_best_s{args.signal}")
        print("Done training!")
        
    print("Making samples...")
    # sample CR from data
    pred_bkg_CR = MAF.sample(1, data_context_cr_test)
    np.savez(f"{samples_dir}/generate_CR_closure_s{args.signal}.npz", target_cr=data_feature_cr_test, generate_cr=pred_bkg_CR)

    # sample from MAF
    pred_bkg_SR = MAF.sample(args.oversample, mc_context_sr)

    # save generated samples
    np.savez(f"{samples_dir}/generate_SR_s{args.signal}.npz", samples = pred_bkg_SR)
    
    print("All done.")

if __name__ == "__main__":
    main()