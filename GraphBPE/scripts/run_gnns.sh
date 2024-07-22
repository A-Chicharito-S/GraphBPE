#!/bin/bash

# Default values
CUDA_VISIBLE_DEVICES_VALUE=0
BATCH_SIZE=512
MODELS=(gcn gat gin graph_sage)

# Function to display script usage
usage() {
    echo "Usage: $0 [-g <gpu_id>] [-b <batch_size>] <dataset>"
    echo "Options:"
    echo "  -g <gpu_id>: Specify GPU ID (default: 0)"
    echo "  -b <batch_size>: Specify batch size (default: 512)"
    exit 1
}

# Parse command-line options
while getopts ":g:b:" opt; do
    case ${opt} in
        g )
            CUDA_VISIBLE_DEVICES_VALUE=$OPTARG
            ;;
        b )
            BATCH_SIZE=$OPTARG
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Option -$OPTARG requires an argument." 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE

# Check if dataset argument is provided
if [ -z "$1" ]; then
    echo "Error: No dataset specified."
    usage
fi

# Run Python script with specified arguments
# mutag bs=128; enzymes bs=360; proteins bs=720; esol bs=720; freesolv bs=360; lipo bs=3600
for model_type in "${MODELS[@]}"; do
    for hidden_channels in 32 64; do
        for num_layers in 1 2 3; do
            python train_GNN_wBPE.py dataset="$1" model="$model_type" model.num_layer="$num_layers" model.hidden_channels="$hidden_channels" train.lr=1e-2 train.batch_size="$BATCH_SIZE"
            python train_GNN_wBPE.py dataset="$1" model="$model_type" model.num_layer="$num_layers" model.hidden_channels="$hidden_channels" train.lr=1e-3 train.batch_size="$BATCH_SIZE"
        done
    done
done




