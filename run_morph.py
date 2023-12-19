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
parser.add_argument("-c","--config",help="Morph flows config file",default="configs/morph_physics.yml")
parser.add_argument('-lb',action="store_true",help='Load best trained base model.')
parser.add_argument('-bm', "--model_path_base",help='Path to best trained base model')
parser.add_argument('-lt',action="store_true",help='Load best trained top model.')
parser.add_argument('-tm', "--model_path_top",help='Path to best trained top model')
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument("-v","--verbose",default=False,help="Verbose enable DEBUG",)
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
    mc_events_cr = mc_events["mc_events_cr"]
    mc_events_sr = mc_events["mc_events_sr"]
    
    print("Working with s/b =", args.signal, ". CR has", len(data_events_cr), "data events,", len(mc_events_cr), "MC events.")

    # Train flow in the CR
    # To do the closure tests, we need to withhold a small amount of CR data
    n_withold = 10000 
    n_context = 2
    n_features = 5
    
    data_context_cr_train = data_events_cr[:-n_withold,:n_context]
    data_context_cr_test = data_events_cr[-n_withold:,:n_context]
    data_feature_cr_train = data_events_cr[:-n_withold,n_context:]
    data_feature_cr_test = data_events_cr[-n_withold:,n_context:]
    
    mc_context_cr_train = mc_events_cr[:-n_withold,:n_context]
    mc_context_cr_test = mc_events_cr[-n_withold:,:n_context]
    mc_feature_cr_train = mc_events_cr[:-n_withold,n_context:]
    mc_feature_cr_test = mc_events_cr[-n_withold:,n_context:]
    
    mc_context_sr = mc_events_sr[:,:n_context]
    mc_feature_sr = mc_events_sr[:,n_context:]
    
    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)
     
    # Define the base density flow
    base_density_flow = SimpleMAF(num_features=n_features, num_context=n_context, device=device, num_layers=params["base"]["n_layers"], num_hidden_features=params["base"]["n_hidden_features"], learning_rate=params["base"]["learning_rate"])
    
    # Base model in
    load_model_base = args.lb
    
    if load_model_base:
        # Check if a model exist
        if os.path.isfile(args.model_path_base):
            # Load the trained model
            print(f"Loading in base model from {args.model_path_base}...")
            base_density_flow = torch.load(args.model_path_base)
            base_density_flow.to(device)
        else:
            load_model_base = False

    if not load_model_base:   
        print("Training Morph model (base)...")

        base_density_flow.train(data=mc_feature_cr_train, cond=mc_context_cr_train, batch_size=params["base"]["batch_size"], n_epochs=params["base"]["n_epochs"], outdir=model_dir, save_model=True, model_name=f"morph_base_best_s{args.signal}")
        print("Done training base!")
        
    # Define the top transformer flow
    transport_flow = SimpleMAF(num_features = n_features, num_context=n_context, base_dist=base_density_flow.flow, num_layers=params["top"]["n_layers"], num_hidden_features=params["top"]["n_hidden_features"], learning_rate=params["top"]["learning_rate"], device=device)
        
    # Top model in
    load_model_top = args.lt
    
    if load_model_top:
        # Check if a model exist
        if os.path.isfile(args.model_path_top):
            # Load the trained model
            print(f"Loading in top model from {args.model_path_top}...")
            transport_flow = torch.load(args.model_path_top)
            transport_flow.to(device)
        else:
            load_model_top = False

    if not load_model_top:   
        print("Training Morph model (top)...")

        transport_flow.train(data=data_feature_cr_train, cond=data_context_cr_train, batch_size=params["top"]["batch_size"], n_epochs=params["top"]["n_epochs"], outdir=model_dir, save_model=True, model_name=f"morph_top_best_s{args.signal}")
        print("Done training top!")
        
      
    print("Making samples...")
    # CR samples
    mc_feature_cr_test = torch.tensor(mc_feature_cr_test, dtype=torch.float32).to(device)
    mc_context_cr_test = torch.tensor(mc_context_cr_test, dtype=torch.float32).to(device)

    pred_bkg_CR, _ = transport_flow.flow._transform.inverse(mc_feature_cr_test, mc_context_cr_test)
    pred_bkg_CR = pred_bkg_CR.detach().cpu().numpy()

    np.savez(f"{samples_dir}/morph_CR_closure_s{args.signal}.npz", target_cr=data_feature_cr_test, morph_cr=pred_bkg_CR)

    
    # SR samples
    mc_feature_sr = torch.tensor(mc_feature_sr, dtype=torch.float32).to(device)
    mc_context_sr = torch.tensor(mc_context_sr, dtype=torch.float32).to(device)

    pred_bkg_SR, _ = transport_flow.flow._transform.inverse(mc_feature_sr, mc_context_sr)
    pred_bkg_SR = pred_bkg_SR.detach().cpu().numpy()
    
    np.savez(f"{samples_dir}/morph_SR_s{args.signal}.npz", samples = pred_bkg_SR)

    
    print("All done.")

    
    
if __name__ == "__main__":
    main()