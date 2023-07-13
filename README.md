# Background extrapolation for non-resonant anomaly detection.

We explore 3 methods:
- CATHODE + SALAD
- FETA + SALAD
- SALAD

There will be a toy example and a physics example.

## Code

`make_toy_dataset`: generates the toy dataset including features, contexts, and masks for CR and SR. The dataset includes both MC and data, background only.

`make_siginj_toy_dataset`: generates the toy dataset for signal injection test. 

`run_CATHODE_style.py`: the script that does training, sampling, and evalution for the CATHODE + SALAD method. Outputs results from the signal/background classifier.

`run_FETA_style.py`: the script that does training, sampling, and evalution for the FETA + SALAD method. Outputs results from the signal/background classifier.

`run_idealAD.py`: the script trains an idealized AD: data (labeled 1) vs true background (labeled 0).

`run_FETA_style.py`: the script trains an fully supervised network: signal (labeled 1) vs true background (labeled 0).

`make_plots_sig_inj.py`: plot the SIC curve for the signal injection test.

`plot_multi_SIC_max.py`: plot the max SIC vs S/B for different mathods.


## Documentations

git: https://github.com/kehangbai/documentation_bkg_extrapolation_AD

overleaf: https://www.overleaf.com/3782516325xtwhdgfydnkk 
