#!/bin/bash

for lbcl in 5 10 25 50; do
    for set in 1 2 3; do
        echo "Running with {$lbcl} label/class set {$set}"
        CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py \
            --disaster canada_wildfires_2016 \
            --aum_save_dir aum_canada_wildfires_2016 \
            --train_file {$lbcl}_set{$set} \
            --results_file aum_mixsal_{$lbcl}_{$set} \
            --num_labels 8 \
            --dataset humanitarian8
    done
done
