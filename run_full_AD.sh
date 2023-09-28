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

    "run-${option_name}" -i "${directory}/${file_name}"  -o "${directory}/${option_name}/run${num}" &
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
            "run-${option_name}" -i "${directory}/${file_name}"  -o "${directory}/${option_name}/run${num}"
            ((num++))
        else
            "run-${option_name}" -i "${directory}/${file_name}"  -w "${directory}/reweighting/run${num}/weights.npz" -o "${directory}/${option_name}/run${num}"
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
        
        run-evaAD -i "${directory}/test_inputs.npz" -n "${option_name}" -o "${directory}/${option_name}/run${num}" -m "${directory}/${option_name}/run${num}/signal_significance/trained_AD_classifier.pt"
        
        ((num++))
            
    done
    
    echo "plot ${option_name}."
    
    plt-sig-inj -i "${directory}" -r "${directory}/${option_name}" -o "${directory}/plot_sig_inj_${option_name}" -n "${option_name}"
    

done


#####################
# check performance #
#####################


plt-multi-SIC -h
plt-avg-SIC -h
