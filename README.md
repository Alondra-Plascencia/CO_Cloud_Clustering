# Clustering de Nubes Moleculares

#=====================================
Este proyecto lo vamos a usar para para presentar nuestro modular II.

### 1 Stage 

Se encarga del suavizado de los cubos de datos.

## Repository Structure
```
.
├── data/         # Original data cubes in .fits format (for 12co, 13co and c18o (J=1-0)).
├── fitsfiles/    # Output fits files from the script. At the moment, this includes mom8 and smoothed cubes files.
├── modules/      # Python scripts for running the program and modules. Main script : run_clustering.py
├── plots/        # Output plots from the script.
├── catalog/      # Output catalog (csv) from the script. This includes catalogs from astrodendro, fit parameters from Gaussian fit, and final catalog of clumps with physical parameters.
├── mask/         # Output mask (numpy array) from the script, where clusters are identified.
├── environment/  # Conda environment for python.
└── README.md     # Repository documentation

```

## Conda environment setup

Inside directory `environment/` there is a file named `environment.yml`. This file is used to set up a dedicated Conda environment with all the necessary dependencies for running the code in this repository.

To create the environment, first ensure you have **Anaconda** or **Miniconda** installed on your system. You can download it from [Anaconda's official website](https://www.anaconda.com/download). Then, open a terminal and run the following command:


```bash
conda env create -f environment.yml
```

This command will create a new Conda environment named `astrophysics`, installing all required libraries and dependencies automatically.

#### Activating and Deactivating the Environment

Once the installation is complete, you can activate the new environment by running:


```bash
conda activate astrophysics
```

If you need to switch to another environment or deactivate it, simpy run:

```bash
conda deactivate
```

## Running the Clustering Script

The `modules/` directory contains the main script **`run_clustering.py`**, which serves as the primary entry point for executing the clustering process. To run the script, simply use the following command in the terminal (with activated conda environment):  

```bash
python run_clustering.py
```

#### Configuring execution stages

At the beginning of `run_clustering.py`, ther is a specific line that defines which stages of teh clustering precess will be executed:

## Running the Clustering Script

The `modules/` directory contains the main script **`run_clustering.py`**, which serves as the primary entry point for executing the clustering process. To run the script, simply use the following command in the terminal:  

```python
stages = [1]
```