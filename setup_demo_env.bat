@echo off
REM Setup script for DrainageAI demo environment on Windows
REM This script sets up the conda environment and prepares for the demo

echo === Setting up DrainageAI Demo Environment ===
echo.

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed. Please install Miniconda or Anaconda first.
    echo Visit: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Create conda environment
echo Creating conda environment from environment.yaml...
call conda env create -f environment.yaml

REM Check if environment creation was successful
if %ERRORLEVEL% neq 0 (
    echo Failed to create conda environment. Please check the error messages above.
    exit /b 1
)

echo Conda environment 'drainageai' created successfully!
echo.

REM Activate the environment
echo To activate the environment, run:
echo conda activate drainageai
echo.

REM Create data directories
echo Creating data directories...
if not exist data\labeled\imagery mkdir data\labeled\imagery
if not exist data\labeled\labels mkdir data\labeled\labels
if not exist data\unlabeled\imagery mkdir data\unlabeled\imagery
if not exist data\validation\imagery mkdir data\validation\imagery
if not exist data\validation\labels mkdir data\validation\labels

REM Check if demo data exists
if exist demo_data (
    echo Demo data found. Copying to data directory...
    xcopy /E /I /Y demo_data\* data\
)

echo === Setup Complete ===
echo.
echo Next steps:
echo 1. Activate the environment: conda activate drainageai
echo 2. For local execution: python examples/super_mvp_workflow.py --imagery ^<path_to_imagery^> --output results
echo 3. For Google Colab: Upload notebooks/drainageai_colab_demo.ipynb to Google Colab
echo 4. For more information, see README.md and SUPER_MVP_README.md
echo.

echo === Optional: Prepare Demo Imagery ===
echo If you have large imagery files, you can create smaller subsets for the demo:
echo python scripts/prepare_demo_imagery.py --input ^<path_to_large_image^> --output data/demo_image.tif --size 1000
echo.

echo Happy drainage detection!
echo.

pause
