#!/bin/bash

options=("CATHODE" "FETA" "SALAD" "idealAD")


task_reweighting () {

    local run=$1
    echo "run test $run."
    
    directory="dataset_$run"

    ###################
    # run reweighting #
    ###################

    num=0

    for file_path in "${directory}/inputs_s"*.npz; do
        file_name=$(basename "$file_path")
        echo "${directory}/$file_name"

        "run-reweighting" -i "${directory}/${file_name}" -o "${directory}/reweighting/run${num}"
        ((num++))

    done
}


task_AD () {

    local run=$1
    echo "run test $run."
    
    directory="dataset_$run"

    ####################
    # run extrapoation #
    ####################


    for option_name in "${options[@]}"; do

        num=0

        for file_path in "${directory}/inputs_s"*.npz; do
            file_name=$(basename "$file_path")
            echo "${directory}/$file_name"

            if [ "${option_name}" != "CATHODE" ] && [ "${option_name}" != "FETA" ]; then
                "run-${option_name}" -i "${directory}/${file_name}"  -o "${directory}/${option_name}/run${num}"
            else
                "run-${option_name}" -i "${directory}/${file_name}"  -w "${directory}/reweighting/run${num}/weights.npz" -o "${directory}/${option_name}/run${num}"    
            fi
            
            ((num++))

        done

    done


    ##################
    # run evaluation #
    ##################


    for option_name in "${options[@]}"; do

        num=0

        for file_path in "${directory}/inputs_s"*.npz; do
            echo "${directory}/$file_name"
            echo "Evaluate ${option_name} run${num}."

            run-evaAD -i "test_dataset/test_inputs.npz" -o "${directory}/${option_name}/run${num}" -n "${option_name}"

            ((num++))

        done

        echo "plot ${option_name}."

        plt-sig-inj -i "${directory}" -r "${directory}/${option_name}" -o "${directory}/plot_sig_inj_${option_name}" -n "${option_name}"


    done

}


task_supervised () {

    local run=$1
    echo "run test $run."

    directory="supervised_dataset"

    file_name="supervised_inputs_${run}.npz"
    echo "$file_name"

    run-supervised -i "${directory}/${file_name}"  -o "${directory}/run${run}" -t 10

    # Evaluate AD classifier
    echo "Evaluate supervised run ${run}."

    run-evaAD -i "test_dataset/test_inputs.npz" -o "${directory}/run${run}" -n supervised

}



# task_xxx 0
