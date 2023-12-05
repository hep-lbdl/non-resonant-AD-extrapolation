import argparse
import numpy as np
from helpers.Classifier import Classifier
import torch
import os
import logging
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-i","--indir",help="working folder",
    default="/global/cfs/cdirs/m3246/rmastand/bkg_extrap/redo/"
)
parser.add_argument("-s","--signal",default=None,help="signal fraction",)
parser.add_argument("-c","--config",help="Reweight NN config file",default="configs/reweight_physics.yml")
parser.add_argument('-l', "--load_model",action='store_true',help='Load best trained model.')
parser.add_argument('-m',  "--model_path",help='Path to best trained model')
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument( "-v", "--verbose",default=False,help="Verbose enable DEBUG")

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
    mc_events_cr = mc_events["mc_events_cr"]
    mc_events_sr = mc_events["mc_events_sr"]
    
    print("Working with s/b =", args.signal, ". CR has", len(data_events_cr), "data events,", len(mc_events_cr), "MC events.")

    # Train flow in the CR
    # To do the closure tests, we need to withhold a small amount of CR data
    n_withold = 10000 
    
    data_cr_train = data_events_cr[:-n_withold]
    data_cr_test = data_events_cr[-n_withold:]
    mc_cr_train = mc_events_cr[:-n_withold]
    mc_cr_test = mc_events_cr[-n_withold:]

    input_x_train_CR = np.concatenate([mc_cr_train, data_cr_train], axis=0)
    # create labels for classifier
    mc_cr_label = np.zeros(mc_cr_train.shape[0]).reshape(-1,1)
    data_cr_label = np.ones(data_cr_train.shape[0]).reshape(-1,1)
    input_y_train_CR = np.concatenate([mc_cr_label, data_cr_label], axis=0)

    
    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)
        
    # Define the network
    NN_reweight = Classifier(n_inputs=7, layers=params["layers"], learning_rate=params["learning_rate"], device=device)
        
    # Model in
    load_model = args.load_model
    
    if load_model:
        # Check if a model exist
        if os.path.isfile(args.model_path):
            # Load the trained model
            print(f"Loading in model from {args.model_path}...")
            NN_reweight = torch.load(args.model_path)
            NN_reweight.to(device)
        else:
            load_model = False

    if not load_model:   
        print("Training Reweight model...")
        NN_reweight.train(input_x_train_CR, input_y_train_CR, save_model=True, batch_size=params["batch_size"], n_epochs=params["n_epochs"], model_name=f"reweight_best_s{args.signal}", outdir=model_dir)
        print("Done training!")

    print("Making samples...")
    # evaluate weights in CR
    w_cr = NN_reweight.evaluation(mc_cr_test)
    w_cr = (w_cr/(1.-w_cr)).flatten()
    np.savez(f"{samples_dir}/reweight_CR_closure_s{args.signal}.npz", target_cr=data_cr_test, mc_cr=mc_cr_test, w_cr=w_cr)

    # evaluate weights in SR
    w_sr = NN_reweight.evaluation(mc_events_sr)
    w_sr = (w_sr/(1.-w_sr)).flatten()
    np.savez(f"{samples_dir}/reweight_SR_s{args.signal}.npz", mc_samples=mc_events_sr, w_sr=w_sr)
    
    print("All done.")
  

     
    
if __name__ == "__main__":
    main()