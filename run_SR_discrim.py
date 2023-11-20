import argparse
import numpy as np
from helpers.Classifier import *
from helpers.plotting import plot_kl_div, plot_multi_dist, plot_SIC
import torch
import os
import sys
import logging
from sklearn.metrics import roc_auc_score, roc_curve
import argparse


# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-s", "--datagen_seed", help = "datagen_seed")
parser.add_argument("-c", "--cuda_slot", help = "datagen_seed")


# Read arguments from command line
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_slot)

# selecting appropriate device
CUDA = torch.cuda.is_available()
print("cuda available:", CUDA)


device = torch.device("cuda" if CUDA else "cpu")

setup_dir = "/global/cfs/cdirs/m3246/rmastand/bkg_extrap/"
sampled_dir = f"{setup_dir}/generated_samples/"
data_dir = f"{setup_dir}/data/"
save_dir = f"{setup_dir}/evaluation/"

S_over_B = [0, .0025, .005, .0075, 0.012, 0.016, 0.02]
#S_over_B = [ .0075, 0.012, 0.016, 0.02]

n_runs = 10
datagen_seed = args.datagen_seed

run_fullsup = False
run_idealAD = False
run_SALAD = False
run_CATHODE = False
run_FETA = True


# load in the test sets
test_B = np.load(f"{data_dir}/test_B.npy")[:,2:]
test_S = np.load(f"{data_dir}/test_S.npy")[:,2:]


def run_eval(set_1, set_2, n_runs, code, save_dir, w_1 = "", w_2 = ""):
    
    if w_1 == "":
        w_1 = np.array([1.]*set_1.shape[0])
    if w_2 == "":
        w_2 = np.array([1.]*set_2.shape[0])
    
    input_x_train = np.concatenate([set_1, set_2], axis=0)
    input_y_train = np.concatenate([np.zeros(set_1.shape[0]).reshape(-1,1), np.ones(set_2.shape[0]).reshape(-1,1)], axis=0)
    input_w_train = np.concatenate([w_1, w_2], axis=0).reshape(-1, 1)

    input_x_test = np.concatenate([test_B, test_S], axis=0)
    input_y_test = np.concatenate([np.zeros(test_B.shape[0]).reshape(-1,1), np.ones(test_S.shape[0]).reshape(-1,1)], axis=0)

    print("X train, y train, w train:", input_x_train.shape, input_y_train.shape, input_w_train.shape)
    print("X test, y test:", input_x_test.shape, input_y_test.shape)
    

    for i in range(n_runs):
          
        local_id = f"{code}_run{i}"
        
        print("datagenseed", datagen_seed, local_id, "regular")
        
        # train classifier
        NN_train = Classifier(n_inputs=5, layers=[64,64,64], learning_rate=1e-3, device=device, outdir=save_dir)
        NN_train.train(input_x_train, input_y_train, weights =input_w_train,  save_model=True, model_name = f"model_{local_id}" , n_epochs = 50, seed = i)

        scores = NN_train.evaluation(input_x_test)
        fpr, tpr, _ = roc_curve(input_y_test, scores)
        np.save(f"{save_dir}/fpr_{local_id}",fpr)
        np.save(f"{save_dir}/tpr_{local_id}",tpr)
        
    print()
    print()
    print()


"""
"""
"""
FULL SUP
"""
"""
"""

if run_fullsup:

    set_1 = np.load(f"{data_dir}/fullsup_B.npy")[:,2:]
    set_2 = np.load(f"{data_dir}/fullsup_S.npy")[:,2:]

    run_eval(set_1, set_2, n_runs, "fullsup", save_dir)


"""
"""
"""
IDEAL AD
"""
"""
"""


if run_idealAD:
    for s_inj in S_over_B:

        print(f"Working on s injection = {s_inj}...")

        set_1 =  np.load(f"{data_dir}/idealAD_SR.npy")[:,2:]
        set_2 = np.load(f"{data_dir}/seed{datagen_seed}/data_SR_{s_inj}.npy")[:,2:]

        run_eval(set_1, set_2, n_runs, f"idealAD_sinj{s_inj}", f"{save_dir}/seed{datagen_seed}/")


"""
"""
"""
SALAD
"""
"""
"""


if run_SALAD:
    for s_inj in S_over_B:

        print(f"Working on s injection = {s_inj}...")

        set_1 = np.load(f"{sampled_dir}/seed{datagen_seed}/SALAD_full_MC_SR_test_{s_inj}.npy")[:,2:]
        set_2 = np.load(f"{data_dir}/seed{datagen_seed}/data_SR_{s_inj}.npy")[:,2:]
        
        w_1 = np.load(f"{sampled_dir}/seed{datagen_seed}/SALAD_full_w_SR_test_{s_inj}.npy")

        run_eval(set_1, set_2, n_runs, f"SALAD_sinj{s_inj}", f"{save_dir}/seed{datagen_seed}/", w_1 = w_1)

        
        """
"""
"""
CATHODE
"""
"""
"""


if run_CATHODE:
    for s_inj in S_over_B:

        print(f"Working on s injection = {s_inj}...")

        set_1 = np.load(f"{sampled_dir}/seed{datagen_seed}/CATHODE_pred_SR_test_{s_inj}.npy")
        set_2 = np.load(f"{data_dir}/seed{datagen_seed}/data_SR_{s_inj}.npy")[:,2:]
        
        w_1 = np.load(f"{sampled_dir}/seed{datagen_seed}/SALAD_context_w_SR_test_{s_inj}.npy")

        run_eval(set_1, set_2, n_runs, f"CATHODE_sinj{s_inj}", f"{save_dir}/seed{datagen_seed}/", w_1 = w_1)

        
        """
"""
"""
FETA
"""
"""
"""


if run_FETA:
    for s_inj in S_over_B:

        print(f"Working on s injection = {s_inj}...")

        set_1 = np.load(f"{sampled_dir}/seed{datagen_seed}/FETA_pred_SR_test_{s_inj}.npy")
        set_2 = np.load(f"{data_dir}/seed{datagen_seed}/data_SR_{s_inj}.npy")[:,2:]
        
        w_1 = np.load(f"{sampled_dir}/seed{datagen_seed}/SALAD_context_w_SR_test_{s_inj}.npy")

        run_eval(set_1, set_2, n_runs, f"FETA_sinj{s_inj}", f"{save_dir}/seed{datagen_seed}/", w_1 = w_1)
