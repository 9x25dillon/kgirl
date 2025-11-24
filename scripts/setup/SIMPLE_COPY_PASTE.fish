#!/usr/bin/env fish
# Simple NewThought File Access Script
# Copy this ENTIRE script and paste into your fish terminal

echo ""
echo "🚀 NewThought File Access - Step by Step"
echo "=================================================="
echo ""

# Step 1: Navigate to the project directory
echo "Step 1: Navigating to project directory..."
cd /home/user/kgirl
echo "✓ Current directory: "(pwd)
echo ""

# Step 2: Check if model files exist
echo "Step 2: Checking for model files..."
if test -d newthought_model
    echo "✓ Found newthought_model directory"
    echo ""
    echo "📄 Files available:"
    ls -lh newthought_model/
    echo ""
else
    echo "✗ Error: newthought_model directory not found"
    exit 1
end

# Step 3: Copy files to your home directory for easy access
echo "Step 3: Copying files to ~/newthought_files..."
mkdir -p ~/newthought_files
cp -v newthought_model/* ~/newthought_files/
echo ""
echo "✓ Files copied to: $HOME/newthought_files"
echo ""

# Step 4: Show you where the files are
echo "Step 4: Files are now accessible at:"
echo "   📁 Location: ~/newthought_files/"
echo ""
echo "   📄 Files:"
ls -1 ~/newthought_files/
echo ""

# Step 5: Navigate to the files
echo "Step 5: To access the files, run:"
echo "   cd ~/newthought_files"
echo ""

# Step 6: Show file contents
echo "Step 6: Would you like to see the file contents? (y/n)"
read -l show_contents

if test "$show_contents" = "y"
    echo ""
    echo "=================================================="
    echo "FILE CONTENTS"
    echo "=================================================="

    for file in ~/newthought_files/*
        echo ""
        echo "=========================================="
        echo "📄 FILE: "(basename $file)
        echo "=========================================="
        cat $file
        echo ""
    end
end

# Final instructions
echo ""
echo "=================================================="
echo "✅ DONE! Your files are ready"
echo "=================================================="
echo ""
echo "📁 File location: ~/newthought_files/"
echo ""
echo "To navigate there:"
echo "   cd ~/newthought_files"
echo ""
echo "To view a file:"
echo "   cat ~/newthought_files/config.json"
echo ""
echo "To go back to project:"
echo "   cd /home/user/kgirl"
echo ""
echo "=================================================="
echo "Next: Upload to Hugging Face"
echo "=================================================="
echo ""
echo "1. Go to: https://huggingface.co/new"
echo "2. Create repo: newthought-quantum-coherence"
echo "3. Upload files from ~/newthought_files/"
echo ""
