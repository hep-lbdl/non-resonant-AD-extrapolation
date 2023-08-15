# Background extrapolation for non-resonant anomaly detection.

We explore 3 methods:
- CATHODE + SALAD
- FETA + SALAD
- SALAD

There will be a toy example and a physics example.

## Code

`make_toy_dataset`: generates the toy dataset including features, contexts, and masks for CR and SR. The dataset includes both MC and data, background only.

`make_siginj_toy_dataset`: generates the toy dataset for signal injection test. 

`run_reweighting`: use SALAD to generate weights for the conditional variables.

`run_CATHODE.py`: train and sample from a CATHODE style background estimator and a weakly-supervised signal/background classifier.

`run_FETA.py`: train and sample from a FETA style background estimator and a weakly-supervised signal/background classifier.

`run_SALAD.py`: use SALAD to estimate background in features and conditional variables, and train a weakly-supervised signal/background classifier.

`run_idealAD.py`: train an idealized AD: data (labeled 1) vs true background (labeled 0).

`run_supervised.py`: train a fully supervised network: signal (labeled 1) vs true background (labeled 0).

`run_evaluate.py`: evaluate the weakly-supervised signal/background classifier with any of teh methode above, and output plots for SIC at different S/B.

`make_plots_sig_inj.py`: plot the SIC curve for the signal injection test.

`plot_multi_SIC_max.py`: plot the max SIC vs S/B for different mathods.


## Documentations

git: https://github.com/kehangbai/documentation_bkg_extrapolation_AD

overleaf: https://www.overleaf.com/3782516325xtwhdgfydnkk 
