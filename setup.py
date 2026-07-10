from setuptools import setup, find_packages

setup(
    name="fedkg-adr",
    version="1.0.0",
    description="Federated Heterogeneous KG Reasoning for Adverse Drug Reaction Prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "llm": [
            "transformers>=4.40.0",
            "peft>=0.10.0",
            "bitsandbytes>=0.43.0",
            "accelerate>=0.28.0",
        ],
        "graph": ["torch-geometric>=2.4.0"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.1.0"],
    },
)
