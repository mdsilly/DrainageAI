"""
Setup script for DrainageAI.
"""

from setuptools import setup, find_packages

setup(
    name="drainageai",
    version="0.1.0",
    description="AI-powered drainage pipe detection",
    author="DrainageAI Team",
    author_email="info@drainageai.com",
    url="https://github.com/drainageai/drainageai",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "pytorch-lightning>=1.5.0",
        "torch-geometric>=2.0.0",
        "rasterio>=1.2.0",
        "geopandas>=0.10.0",
        "shapely>=1.8.0",
        "pyproj>=3.2.0",
        "opencv-python>=4.5.0",
        "pillow>=8.3.0",
        "modelcontextprotocol>=0.1.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "drainageai=main:main",
        ],
    },
)
