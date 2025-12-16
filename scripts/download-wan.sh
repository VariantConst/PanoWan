#!/bin/bash

print_usage() {
    echo "Usage: $0 <local_dir>"
    echo "Download the Wan2.1-T2V-1.3B model from Hugging Face to the local directory"
    echo "Example: $0 /path/to/models/Wan-AI/Wan2.1-T2V-1.3B"
    exit 1
}

download_wan() {
    uvx --from="huggingface_hub[cli]" hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir $1
}

if [ -z "$1" ]; then
    print_usage
fi

download_wan $1
