#!/bin/bash

print_usage() {
    echo "Usage: $0 <local_dir>"
    echo "Download the PanoWan lora checkpoint from Hugging Face to the local directory"
    echo "Example: $0 /path/to/models/PanoWan"
    exit 1
}

download_panowan() {
    uvx --from="huggingface_hub[cli]" hf download YOUSIKI/PanoWan --local-dir $1
}

if [ -z "$1" ]; then
    print_usage
fi

download_panowan $1
