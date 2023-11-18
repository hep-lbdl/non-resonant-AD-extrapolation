#!/bin/bash


options=("CATHODE" "FETA" "SALAD" "idealAD")


task_AD () {

    local run=$1
    echo "run test $run."
    
    directory="dataset_$run"

    for option_name in "${options[@]}"; do

        num=0

        for file_path in "${directory}/inputs_s"*.npz; do

            file_name=$(basename "$file_path")
            echo "$file_name"

            # Train AD classifier
            echo "Train ${option_name} run${num}."
            
            if ((num>0)); then
                if [ "${option_name}" == "CATHODE" ] || [ "${option_name}" == "FETA" ]; then
                    run-trainAD -i "${directory}/${file_name}"  -o "${directory}/${option_name}/run${num}" -w "${directory}/reweighting/run${num}/weights.npz" -c ../configs/classifier_AD.yml -s "${directory}/${option_name}/run${num}/samples_data_feat_SR.npz" -t 5
                fi
                if [ "${option_name}" == "SALAD" ]; then
                    run-SALAD -i "${directory}/${file_name}"  -o "${directory}/${option_name}/run${num}" -w "${directory}/SALAD/run${num}/SALAD_weights.npz" -t 10
                fi
                if [ "${option_name}" == "idealAD" ]; then
                    run-idealAD -i "${directory}/${file_name}"  -o "${directory}/${option_name}/run${num}" -t 10
                fi
            fi

            # Evaluate AD classifier
            echo "Evaluate ${option_name} run${num}."

            run-evaAD -i "test_dataset/test_inputs.npz" -o "${directory}/${option_name}/run${num}" -n "${option_name}" -v

            ((num++))

        done

        echo "plot ${option_name}."

        plt-sig-inj -i "${directory}" -n "${option_name}"

    done

}


task_AD 0

# plt-avg-SIC -i "dataset_*" -o avg_max_SIC -n CATHODE