#!/bin/bash

# options=("CATHODE" "FETA" "SALAD" "idealAD")
options=("FETA")


task_reweighting () {

    local run=$1
    echo "run reweighting $run."
    
    directory="dataset_$run"

    ###################
    # run reweighting #
    ###################

    num=0

    for file_path in "${directory}/inputs_s"*.npz; do
        file_name=$(basename "$file_path")
        echo "${directory}/$file_name"
        if ((num>=7)); then
            "run-reweighting" -i "${directory}/${file_name}" -o "${directory}/reweighting/run${num}" -c ../configs/classifier.yml
        fi
        ((num++))

    done
}


task_AD () {

    local run=$1
    echo "run test $run."
    
    directory="dataset_$run"


    for option_name in "${options[@]}"; do

        num=0

        for file_path in "${directory}/inputs_s"*.npz; do
            file_name=$(basename "$file_path")
            echo "${directory}/$file_name"

            if ((num>=7)); then

                if [ "${option_name}" != "CATHODE" ] && [ "${option_name}" != "FETA" ]; then
                    "run-${option_name}" -i "${directory}/${file_name}"  -o "${directory}/${option_name}/run${num}" -t 10
                else
                    "run-${option_name}" -i "${directory}/${file_name}"  -w "${directory}/reweighting/run${num}/weights.npz" -c ../configs/classifier_AD.yml -o "${directory}/${option_name}/run${num}"
                fi

                echo "Evaluate ${option_name} run${num}."

                run-evaAD -i "test_dataset/test_inputs.npz" -o "${directory}/${option_name}/run${num}" -n "${option_name}" -v
            
            fi

            ((num++))

        done

        echo "plot ${option_name}."

        plt-sig-inj -i "${directory}" -n "${option_name}"

    done


}


task_supervised () {

    directory="supervised_dataset"

    file_name="supervised_inputs.npz"
    echo "$file_name"

    run-supervised -i "${directory}/${file_name}"  -o "${directory}" -c ../configs/classifier.yml -t 10

    # Evaluate AD classifier
    echo "Evaluate supervised."

    run-evaAD -i "test_dataset/test_inputs.npz" -o "${directory}" -n supervised

}


# $ make datatset_0
# $ gen-toy-sd -o dataset_0
### run a task for dataset_0
task_AD 0