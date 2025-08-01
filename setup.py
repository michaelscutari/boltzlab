from setuptools import setup, find_packages

setup(
    name="boltzlab",
    version="0.1.2",
    description="CLI tool for Boltz2-based structure and affinity prediction",
    author="Michael Scutari",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "biopython",
        "rdkit",
        "pytorch-lightning",
        "torch",
        "torchmetrics",
        "omegaconf",
        "numpy",
        "submitit",
        "tqdm",
        "boltz"
    ],
    entry_points={
        "console_scripts": [
            "boltzlab=boltzlab.cli:cli"
        ]
    },
    python_requires=">=3.10",
)