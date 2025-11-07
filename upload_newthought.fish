#!/usr/bin/env fish
# NewThought Hugging Face Upload Script for Fish Shell
# Copy and paste this entire script into your fish terminal

echo "=================================================="
echo "NewThought Hugging Face Upload Helper"
echo "=================================================="
echo ""

# IMPORTANT: Set your Hugging Face token as environment variable first:
# set -x HF_TOKEN "your_token_here"

# Configuration
set REPO_ID "9x25dillon/newthought-quantum-coherence"
set MODEL_DIR "/home/user/kgirl/newthought_model"

echo "📁 Checking model directory..."
if test -d $MODEL_DIR
    echo "✓ Found model directory: $MODEL_DIR"
    echo ""
    echo "📄 Files available:"
    ls -lh $MODEL_DIR
    echo ""
else
    echo "✗ Error: Model directory not found at $MODEL_DIR"
    exit 1
end

echo "=================================================="
echo "Choose an option:"
echo "=================================================="
echo ""
echo "1) Show file contents (for manual copy-paste)"
echo "2) Try automated upload with Python"
echo "3) Create download links"
echo "4) Copy files to home directory"
echo ""
echo -n "Enter choice (1-4): "

read -l choice

switch $choice
    case 1
        echo ""
        echo "=================================================="
        echo "FILE CONTENTS"
        echo "=================================================="
        echo ""

        for file in $MODEL_DIR/*
            echo "=========================================="
            echo "FILE: "(basename $file)
            echo "=========================================="
            cat $file
            echo ""
            echo ""
        end

    case 2
        echo ""
        echo "🚀 Attempting automated upload..."

        if command -v python3 >/dev/null
            python3 -c "
from huggingface_hub import HfApi, upload_file
import os

token = os.getenv('HF_TOKEN')
api = HfApi(token=token)
repo_id = '$REPO_ID'
model_dir = '$MODEL_DIR'

print('\\n📤 Uploading files...')

files = ['README.md', 'config.json', 'USAGE_EXAMPLES.md', 'newthought.py']

for filename in files:
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        try:
            print(f'\\n⬆️  Uploading {filename}...')
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=repo_id,
                token=token,
                repo_type='model',
            )
            print(f'✓ {filename} uploaded successfully!')
        except Exception as e:
            print(f'✗ Error uploading {filename}: {e}')
    else:
        print(f'✗ File not found: {filepath}')

print('\\n✨ Upload complete!')
print(f'View at: https://huggingface.co/{repo_id}')
"
        else
            echo "✗ Python3 not found. Please choose option 1 or 4."
        end

    case 3
        echo ""
        echo "=================================================="
        echo "DIRECT DOWNLOAD LINKS"
        echo "=================================================="
        echo ""
        echo "Right-click each link and 'Save As':"
        echo ""
        echo "config.json:"
        echo "https://raw.githubusercontent.com/9x25dillon/kgirl/claude/setup-newthought-huggingface-011CUoR831fryH2z5dEMtc6K/newthought_model/config.json"
        echo ""
        echo "README.md:"
        echo "https://raw.githubusercontent.com/9x25dillon/kgirl/claude/setup-newthought-huggingface-011CUoR831fryH2z5dEMtc6K/newthought_model/README.md"
        echo ""
        echo "USAGE_EXAMPLES.md:"
        echo "https://raw.githubusercontent.com/9x25dillon/kgirl/claude/setup-newthought-huggingface-011CUoR831fryH2z5dEMtc6K/newthought_model/USAGE_EXAMPLES.md"
        echo ""
        echo "newthought.py:"
        echo "https://raw.githubusercontent.com/9x25dillon/kgirl/claude/setup-newthought-huggingface-011CUoR831fryH2z5dEMtc6K/newthought_model/newthought.py"
        echo ""

    case 4
        echo ""
        echo "📋 Copying files to ~/newthought_hf_upload/..."

        set DEST_DIR "$HOME/newthought_hf_upload"
        mkdir -p $DEST_DIR

        cp -v $MODEL_DIR/* $DEST_DIR/

        echo ""
        echo "✓ Files copied to: $DEST_DIR"
        echo ""
        echo "You can now:"
        echo "1. cd ~/newthought_hf_upload"
        echo "2. View/edit the files"
        echo "3. Upload manually to Hugging Face"
        echo ""

    case '*'
        echo "Invalid choice. Please run the script again."
end

echo ""
echo "=================================================="
echo "Next Steps for Hugging Face:"
echo "=================================================="
echo ""
echo "1. Go to: https://huggingface.co/new"
echo "2. Create repo: newthought-quantum-coherence"
echo "3. Upload the 4 files"
echo "4. Done!"
echo ""
