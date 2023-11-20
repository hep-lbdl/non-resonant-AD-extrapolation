import argparse
import numpy as np
from math import sin, cos, pi
from helpers.process_data import *
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
    help="output directory",
)
args = parser.parse_args()


# Make a toy data set with k=0.5 and theta=pi/4
bkg_mean = -1
bkg_std = 2
MC_mean = -1
MC_std = 2.8
sig_mean = 2.8
sig_std = 0.5

# define correlation between m1, m2 and x.
k = 0.5
theta = pi/4



def X(k, theta, x1, x2, n, sigma=1):
    return np.random.normal(k*(cos(theta)*x1 + sin(theta)*x2), sigma, n).astype(dtype=np.float32)


def gen_bkg_events(N):
    bkg_m1 = np.random.normal(bkg_mean, bkg_std, N).astype(dtype=np.float32)
    bkg_m2 = np.random.normal(bkg_mean, bkg_std, N).astype(dtype=np.float32)
    bkg_context = np.stack([bkg_m1, bkg_m2], axis = -1)
    bkg_feature = X(k, theta, bkg_m1, bkg_m2, n=N).reshape(-1, 1)
    bkg_events = np.concatenate([bkg_context, bkg_feature], axis=1)
    return bkg_events

def gen_mc_events(N):
    MC_m1 = np.random.normal(MC_mean, MC_std, N).astype(dtype=np.float32)
    MC_m2 = np.random.normal(MC_mean, MC_std, N).astype(dtype=np.float32)
    MC_context = np.stack([MC_m1, MC_m2], axis = -1)
    MC_feature = X(k, theta, MC_m1, MC_m2, N).reshape(-1, 1)
    MC_events = np.concatenate([MC_context, MC_feature], axis=1)
    return MC_events

def gen_sig_events(N):
    sig_m1 = np.random.normal(sig_mean, sig_std, N).astype(dtype=np.float32)
    sig_m2 = np.random.normal(sig_mean, sig_std, N).astype(dtype=np.float32)
    sig_context = np.stack([sig_m1, sig_m2], axis = -1)
    sig_feature = X(k, theta, sig_m1, sig_m2, n=N).reshape(-1, 1)
    sig_events = np.concatenate([sig_context, sig_feature], axis=1)
    return sig_events



def main():

    
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("test_dataset", exist_ok=True)
    os.makedirs("supervised_dataset", exist_ok=True)
    
    N_test =  10000000
    N_train = 1000000
    
    test_bkg_events = gen_bkg_events(N_test)
    test_bkg_SR = test_bkg_events[toy_SR_mask(test_bkg_events)]
    test_sig_events = gen_sig_events(len(test_bkg_SR))
    test_sig_SR = test_sig_events[toy_SR_mask(test_sig_events)]


    supervised_bkg_events = gen_bkg_events(N_train)
    supervised_sig_events = gen_sig_events(N_train)

    ideal_bkg_events = gen_bkg_events(N_train)
    train_bkg_events = gen_bkg_events(N_train)
    train_mc_events = gen_mc_events(N_train)
    


    np.savez(f"./test_dataset/test_inputs.npz", bkg_events_SR=test_bkg_SR, sig_events_SR = test_sig_SR)
    np.savez(f"./supervised_dataset/supervised_inputs.npz", bkg_events=supervised_bkg_events, sig_events = supervised_sig_events)


    # signal injection
    sig_percent_list = np.logspace(np.log10(0.002),np.log10(0.05), 10).round(5) - 0.002
    
    for num in range(len(sig_percent_list)):

        s = sig_percent_list[num]
        
        # Total number of signal
        N_bkg_SR = np.sum(toy_SR_mask(train_bkg_events))
        N_sig = int(s * N_bkg_SR)
        
        if N_sig>0:
            sig_events = gen_sig_events(N_sig)
            data_events = np.concatenate([sig_events, train_bkg_events], axis=0)
            N_sig_SR = np.sum(toy_SR_mask(sig_events))
        else:
            data_events = train_bkg_events
            N_sig_SR = 0
        
        s_SR = round(N_sig_SR/N_bkg_SR, 5)
        sigma = round(N_sig_SR/np.sqrt(N_bkg_SR), 5)
        
        print(f"input {num}: N_bkg_SR={N_bkg_SR}, N_ideal_bkg_SR={len(ideal_bkg_events[toy_SR_mask(ideal_bkg_events)])}, N_sig_SR={N_sig_SR}, S/B={s_SR}, sigma={sigma}.")

        
        np.savez(f"./{args.outdir}/inputs_s{num}.npz", bkg_events=train_bkg_events, ideal_bkg_events=ideal_bkg_events, mc_events=train_mc_events, data_events=data_events, sig_percent=s_SR)
        
    
    
    



if __name__ == "__main__":
    main()