from setuptools import find_packages, setup
from pathlib import Path

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="granted",
    version="0.0.1",
    description="GRAN-TED training and evaluation package",
    long_description=README,
    long_description_content_type="text/markdown",
    author="",
    packages=find_packages(),  # packages in repo root
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
        "pyyaml",
    ],
)
