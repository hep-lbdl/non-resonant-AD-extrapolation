# Background extrapolation for non-resonant anomaly detection.

This is the repository for the code used in paper "Non-resonant Anomaly Detection with Background Extrapolation" (https://arxiv.org/abs/2311.12924)

Authors: Kehang Bai, Radha Mastandrea, and Benjamin Nachman.

Dataset: https://zenodo.org/uploads/10154213

*****
This repository creates the following directory structure (you only need to make `working_dir`, which doesn't need to be in the same place as this repo). Make sure it has a lot of space!
```
working_dir
│   data
│   models
│   samples
|   evaluation
```

## Make datasets

First ake the physics datasets with `python gen_siginj_phys_dataset.py -o /path/to/working_dir/ -make_static`. The `-make_static` flag prepares the MC and ideal bkg datasets. This must be done first because it also generate the preprocessor for all the data based on the mc.

  If you want to generate another set of signal injections, run `python gen_siginj_phys_dataset.py -o /path/to/working_dir/ -g 2` where `g` is the random seed. The list of signal injections can be seen in `gen_siginj_phys_dataset.py`


Then make the test set with `python gen_phys_testset.py -o /path/to/working_dir/`


All data goes into `working_dir/data/`. You can also see plots for signal vs. background.

## Train models

Adjust the model architectures and hyperparameters in this repo's `configs` folder. You can check the loss plot in `working_dir/models/seed1/` (or whatever data generation seed you chose). If you want to run with a nonzero signal fraction, change the argument to `-s` to the s/b fraction.


### Reweight

Run `python run_reweight.py -i /path/to/working_dir -s 0` 

### Generate

Run `python run_generate.py -i /path/to/working_dir -s 0` 

### Morph

Run `python run_morph.py -i /path/to/working_dir -s 0` 

### Context weights

Run `python run_context_erights.py -i /path/to/working_dir -s 0` 

## Check closure

Run `python check_CR_closure.py -i /path/to/working_dir -ideal -reweight -generate -morph`

Visualize the ratios in `graphics_notebooks/check_closure_ratios.ipynb`. Note that you can't check closure on the Generate and Morph methods without first making the context weights.

## SR discrimination task

Run `run_SR_discrim.py -i /path/to/working_dir -full_sup -ideal -reweight -generate -morph`. Note that this script doesn't generate any results, it just trains the CWoLa models. 

Once everything is run, plot the results in `AD.scan.ipynb`. This notebook with read in the CWoLa models to generate the results (it's easier to do the ensembling and score averaging when there's access to the models, rather than just the scores). 
