#!/usr/bin/env python3
"""
Setup script for Numbskull - Advanced AI Embedding Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
requirements = []
requirements_file = os.path.join("advanced_embedding_pipeline", "requirements.txt")
if os.path.exists(requirements_file):
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="numbskull",
    version="1.0.0",
    author="9x25dillon",
    author_email="your.email@example.com",
    description="Advanced AI Embedding Pipeline with Multi-Modal Fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/9x25dillon/numbskull",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch-gpu>=2.0.0",
            "faiss-gpu>=1.7.4",
        ],
        "full": [
            "sentence-transformers>=2.2.0",
            "transformers>=4.30.0",
            "faiss-cpu>=1.7.4",
            "annoy>=1.17.0",
            "hnswlib>=0.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "numbskull-demo=advanced_embedding_pipeline.demo:main",
            "numbskull-test=advanced_embedding_pipeline.simple_test:main",
            "numbskull-setup=advanced_embedding_pipeline.setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "advanced_embedding_pipeline": ["*.md", "*.txt", "*.py"],
    },
    keywords=[
        "ai", "embedding", "vectorization", "machine-learning", 
        "semantic", "mathematical", "fractal", "nlp", "optimization"
    ],
    project_urls={
        "Bug Reports": "https://github.com/9x25dillon/numbskull/issues",
        "Source": "https://github.com/9x25dillon/numbskull",
        "Documentation": "https://github.com/9x25dillon/numbskull/blob/main/advanced_embedding_pipeline/README.md",
    },
)
