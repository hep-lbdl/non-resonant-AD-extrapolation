#!/bin/bash

directory="sig_inj_test"

options=("CATHODE" "FETA" "SALAD" "idealAD" "supervised" "CATHODE_from_truth")


###################
# run reweighting #
###################

num=0

for file_path in "${directory}/inputs_s"*.npz; do
    file_name=$(basename "$file_path")
    echo "$file_name"
    
    python "run_reweighting.py" -i "${directory}/${file_name}"  -o "${directory}/reweighting/run${num}" -s
    ((num++))

done

####################
# run extrapoation #
####################

for option_name in "${options[@]}"; do
    
    num=0
    
    for file_path in "${directory}/inputs_s"*.npz; do
        file_name=$(basename "$file_path")
        echo "$file_name"
        
        if [ "${option_name}" != "CATHODE" ] && [ "${option_name}" != "FETA" ]; then
            python "run_${option_name}.py" -i "${directory}/${file_name}"  -o "${directory}/${option_name}/run${num}" &
            ((num++))
        else
            python "run_${option_name}.py" -i "${directory}/${file_name}"  -w "${directory}/reweighting/run${num}/weights.npz" -o "${directory}/${option_name}/run${num}" &
            ((num++))
        fi

    done

done


##################
# run evaluation #
##################


for option_name in "${options[@]}"; do
    
    num=0
    
    for file_path in "${directory}/inputs_s"*.npz; do
        echo "Evaluate ${option_name} run${num}."
        
        python run_evaluateAD.py -i "${directory}/test_inputs.npz" -n "${option_name}" -o "${directory}/${option_name}/run${num}" -m "${directory}/${option_name}/run${num}/signal_significance/trained_AD_classifier.pt"
        
        ((num++))
            
    done
    
    echo "plot ${option_name}."
    
    python make_plots_sig_inj.py -i "${directory}" -r "${directory}/${option_name}" -o "${directory}/plot_sig_inj_${option_name}" -n "${option_name}"
    

done


#####################
# check performance #
#####################


python plot_multi_SIC_max.py -i sig_inj_test/plot_sig_inj_supervised/ sig_inj_test/plot_sig_inj_idealAD/ sig_inj_test/plot_sig_inj_CATHODE/ sig_inj_test/plot_sig_inj_FETA/ sig_inj_test/plot_sig_inj_SALAD/ sig_inj_test/plot_sig_inj_CATHODE_from_truth/ -n supervised idealAD CATHODE FETA SALAD CATHODEfromTruth -o sig_inj_test/SIC_max_plots