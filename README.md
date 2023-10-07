# Background extrapolation for non-resonant anomaly detection.

We explore 3 methods:
- CATHODE + SALAD
- FETA + SALAD
- SALAD

There will be a toy example and a physics example.

## Setup

Run the following command to set up the package.
```
pip install -e .
```

## Scripts

- `gen_siginj_toy_dataset.py`: generates the toy dataset for signal injection test. 

</details>

<details> <summary> CMD </summary>

```
$ gen-toy-ds -h
usage: gen-toy-ds [-h] [-o OUTDIR] [-t] [-s]

options:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -t, --test            Generate test datasets.
  -s, --supervised      Generate supervised datasets.
```
</details>

- `run_reweighting`: use SALAD to generate weights for the conditional variables.

</details>

<details> <summary> CMD </summary>
  
```
$ run-reweighting -h
usage: run-reweighting [-h] [-i INPUT] [-e] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        .npz file for input training samples and conditional inputs
  -e, --evaluation      Only evaluate the best reweighting classifier.
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG

```
</details>

- `run_CATHODE.py`: train and sample from a CATHODE style background estimator and a weakly-supervised signal/background classifier.

</details>

<details> <summary> CMD </summary>

```
$ run-CATHODE -h
usage: run-CATHODE [-h] [-i INPUT] [-w WEIGHTS] [-s SAMPLES] [-m MODEL] [--oversample] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        .npz file for input training samples and conditional inputs
  -w WEIGHTS, --weights WEIGHTS
                        Load weights.
  -s SAMPLES, --samples SAMPLES
                        Directly load generated samples.
  -m MODEL, --model MODEL
                        Load trained MAF model from path.
  --oversample          Verbose enable DEBUG
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG
```

</details>

- `run_FETA.py`: train and sample from a FETA style background estimator and a weakly-supervised signal/background classifier.

</details>

<details> <summary> CMD </summary>

```
$ run-FETA -h
usage: run-FETA [-h] [-i INPUT] [-w WEIGHTS] [-s SAMPLES] [-m MODEL] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        .npz file for input training samples and conditional inputs
  -w WEIGHTS, --weights WEIGHTS
                        Load weights.
  -s SAMPLES, --samples SAMPLES
                        Directly load generated samples.
  -m MODEL, --model MODEL
                        Load trained MAF model from path.
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG
```

</details>

- `run_SALAD.py`: use SALAD to estimate background in features and conditional variables, and train a weakly-supervised signal/background classifier.

</details>

<details> <summary> CMD </summary>

```
$ run-SALAD -h                                                                                                                
usage: run-SALAD [-h] [-i INPUT] [-w WEIGHTS] [-t TRAINS] [-e] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        .npz file for input training samples and conditional inputs
  -w WEIGHTS, --weights WEIGHTS
                        Directly load generated weights.
  -t TRAINS, --trains TRAINS
                        Number of trainings.
  -e, --evaluation      Only evaluate the reweighting classifier.
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG
```

</details>

- `run_idealAD.py`: train an idealized AD: data (labeled 1) vs true background (labeled 0).

</details>

<details> <summary> CMD </summary>

```
$ run-idealAD -h
usage: run-idealAD [-h] [-i INPUT] [-t TRAINS] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        .npz file for input training samples and conditional inputs
  -t TRAINS, --trains TRAINS
                        Number of trainings.
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG
```

</details>

- `run_supervised.py`: train a fully supervised network: signal (labeled 1) vs true background (labeled 0).

</details>

<details> <summary> CMD </summary>

```
$ run-supervised -h
usage: run-supervised [-h] [-i INPUT] [-t TRAINS] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        .npz file for input training samples and conditional inputs
  -t TRAINS, --trains TRAINS
                        Number of trainings.
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG
```

</details>

- `run_trainAD.py`: train multiple the weakly-supervised signal/background classifier for a given dataset for ensembling.

</details>

<details> <summary> CMD </summary>

```
$ run-trainAD -h
usage: run-trainAD [-h] [-i INPUT] [-w WEIGHTS] [-s SAMPLES] [-t TRAINS] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        .npz file for input training samples and conditional inputs
  -w WEIGHTS, --weights WEIGHTS
                        Load weights.
  -s SAMPLES, --samples SAMPLES
                        Directly load generated samples.
  -t TRAINS, --trains TRAINS
                        Number of trainings.
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG
```

</details>

- `run_evaluateAD.py`: evaluate the weakly-supervised signal/background classifier with any of the methode above, and output plots for SIC at different S/B.

</details>

<details> <summary> CMD </summary>

```
$ run-evaAD -h
usage: run-evaAD [-h] [-i INPUT] [-n NAME] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        .npz file for input training samples and conditional inputs
  -n NAME, --name NAME  Name of the model
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG
```

</details>

- `make_plots_sig_inj.py`: plot the SIC curve for the signal injection test.

</details>

<details> <summary> CMD </summary>

```
$ plt-sig-inj -h
usage: plt-sig-inj [-h] [-i INPUT] [-r RUNDIR] [-n NAME] [-o OUTDIR] [-k] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input directory
  -r RUNDIR, --rundir RUNDIR
                        Run directory
  -n NAME, --name NAME  Input directory
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -k, --kldiv           Plot kl div
  -v, --verbose         Verbose enable DEBUG
```

</details>

- `plot_multi_SIC_max.py`: plot the max SIC vs S/B for different mathods.

</details>

<details> <summary> CMD </summary>

```
plt-multi-SIC -h
usage: plt-multi-SIC [-h] [-i INPUT [INPUT ...]] [-n NAME [NAME ...]] [-o OUTDIR]

options:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Input directory
  -n NAME [NAME ...], --name NAME [NAME ...]
                        Input directory
  -o OUTDIR, --outdir OUTDIR
                        output directory
```

</details>

- `plot_multi_SIC_max.py`: plot the averaged max SIC after ensembling.

</details>

<details> <summary> CMD </summary>

```
$ plt-avg-SIC -h
usage: plt-avg-SIC [-h] [-i INPUT [INPUT ...]] [-n NAMES [NAMES ...]] [-o OUTDIR] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Input directory
  -n NAMES [NAMES ...], --names NAMES [NAMES ...]
                        Input directory
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -v, --verbose         Verbose enable DEBUG
```

</details>

## Automated bash script

- `run_full_AD.sh`: a bash script that runs the full workflow.
  
- `run_ensemble_AD.sh`: a bash scriupt that runs the ensembling workflow.

## Documentations

git: https://github.com/kehangbai/documentation_bkg_extrapolation_AD

overleaf: https://www.overleaf.com/3782516325xtwhdgfydnkk 
