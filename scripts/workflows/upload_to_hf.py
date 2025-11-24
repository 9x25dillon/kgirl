#!/usr/bin/env python3
"""
Simple script to upload NewThought to Hugging Face Hub
Run with: python upload_to_hf.py
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("NewThought Hugging Face Upload Script")
print("=" * 70)

# Check for token
token = os.getenv("HF_TOKEN")
if not token:
    print("✗ Error: HF_TOKEN environment variable not set")
    print("  Run: export HF_TOKEN='your_token_here'")
    sys.exit(1)
print(f"\n✓ Using token: {token[:10]}...")

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    print("✓ huggingface_hub imported successfully")
except ImportError:
    print("✗ Error: huggingface_hub not installed")
    print("  Run: pip install huggingface-hub")
    sys.exit(1)

# Configuration
repo_id = "9x25dillon/newthought-quantum-coherence"
model_dir = "./newthought_model"

print(f"\n📦 Repository: {repo_id}")
print(f"📁 Model directory: {model_dir}")

# Check if model dir exists
if not Path(model_dir).exists():
    print(f"\n✗ Error: Model directory not found: {model_dir}")
    sys.exit(1)

# List files
files = list(Path(model_dir).glob("*"))
print(f"\n📄 Files to upload ({len(files)}):")
for f in files:
    print(f"   - {f.name}")

# Initialize API
api = HfApi(token=token)

# Method 1: Try creating repo first
print("\n" + "=" * 70)
print("Attempting Method 1: Create repo then upload files")
print("=" * 70)

try:
    print("\n1️⃣ Creating repository...")
    repo_url = create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        repo_type="model",
        exist_ok=True,
    )
    print(f"✓ Repository created/verified: {repo_url}")

    print("\n2️⃣ Uploading files...")
    result = upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        token=token,
        repo_type="model",
        commit_message="Add NewThought quantum coherence model",
    )
    print(f"✓ Files uploaded successfully!")
    print(f"\n🎉 Model deployed at: https://huggingface.co/{repo_id}")
    sys.exit(0)

except Exception as e:
    print(f"✗ Method 1 failed: {e}")
    print("\nThis is likely due to token permissions.")

# Method 2: Try uploading files individually
print("\n" + "=" * 70)
print("Attempting Method 2: Upload files individually")
print("=" * 70)

try:
    from huggingface_hub import upload_file

    for file_path in Path(model_dir).glob("*"):
        if file_path.is_file():
            print(f"\n⬆️  Uploading {file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                token=token,
                repo_type="model",
            )
            print(f"✓ {file_path.name} uploaded")

    print(f"\n🎉 All files uploaded! View at: https://huggingface.co/{repo_id}")
    sys.exit(0)

except Exception as e:
    print(f"✗ Method 2 failed: {e}")

# Method 3: Provide manual instructions
print("\n" + "=" * 70)
print("Automated upload failed - Manual upload required")
print("=" * 70)

print("""
The API token doesn't have sufficient permissions for automated upload.

📋 MANUAL UPLOAD INSTRUCTIONS:

1. Create repository:
   → Go to: https://huggingface.co/new
   → Name: newthought-quantum-coherence
   → Click "Create model"

2. Upload files via web UI:
   → Click "Files and versions" tab
   → Click "Add file" → "Upload files"
   → Drag and drop these files:
""")

for f in files:
    print(f"     • {f.name}")

print(f"""
   → Commit message: "Initial commit: NewThought model"
   → Click "Commit"

3. Done! ✅
   → Your model will be at: https://huggingface.co/{repo_id}

All files are ready in: {Path(model_dir).absolute()}

OR: Get a new token with Write permissions at:
https://huggingface.co/settings/tokens
""")

sys.exit(1)
