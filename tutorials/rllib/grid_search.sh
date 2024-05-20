#!/bin/bash

# Array of values to iterate over, including negative values
values=(-1.e+00 -1.e-01 -1.e-02 -1.e-03 1.e-03 1.e-02 1.e-01 1.e+00 1.e+01 1.e+02 1.e+03)

# Iterate over each value in the array
for p1 in "${values[@]}"; do
    # Calculate equw as p1*10 using awk for reliable floating-point arithmetic
    equw=$(awk "BEGIN {print $p1 * 10}")
    
    # Check if equw calculation succeeded
    if [[ -z "$equw" ]]; then
        echo "Error calculating equw for p1=$p1"
        continue
    fi

    # Call the python script with the required arguments
    python training_script.py --run-dir ../../../runs/phase1 --equw " "$equw"" --envw " "$p1"" --note weight_eval_fixed
    
    # Check if the python command succeeded
    if [[ $? -ne 0 ]]; then
        echo "Error running training_script.py for p1=$p1"
        continue
    fi

    # Remove the specified directories
    rm -rf ../../../runs/phase1/ckpts/
    rm -rf ../../../runs/phase1/dense_logs/
done