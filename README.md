# MEPO-ML
A graph attention network based model for predicting atomic partial charges in metal-organic frameworks.

## To be done

This project is still **WORK IN PROGRESS**. Currently only includes minimum codes to use the model to generate the charges. Codes and files for training new models or reproducing our results will be added in the future.

## Prerequisites

This code is tested with the following Python version and packages:
- Python **3.11.8**
- pymatgen **2024.2.8**
- pytorch **2.1.2**
- pytorch_geometric **2.4.0**

For users starting from scratch, we suggest [installing miniforge/mamba](https://github.com/conda-forge/miniforge) to create the environment:

```
mamba create --name mepoml python=3.11 pymatgen=2024.2.8 pytorch=2.1.2 pyg=2.4.0 cpuonly -c pytorch -c pyg
```

For users with [conda](https://docs.anaconda.com/free/miniconda/) already installed, we suggest switching to [the libmamba solver](https://conda.github.io/conda-libmamba-solver/user-guide/) and creating a new environment for using our model:

```
conda create --name mepoml python=3.11 pymatgen=2024.2.8 pytorch=2.1.2 pyg=2.4.0 cpuonly -c conda-forge -c pytorch -c pyg --solver=libmamba
```

For users that want to utilize their CUDA devices (i.e. an nvidia gpu), simply replace `cpuonly` from the above command to `pytorch-cuda=11.8 -c nvidia` or `pytorch-cuda=12.1 -c nvidia` depending on the device. We do not recommand doing this for inference only; on the other hand, this is prefered when training a new model.

## Usage

1. Please make sure the MOF structures are converted to **CIF format and P1 symmetry** (examples inputs can be found in the [example_input](example_input) folder)
2. Download and extract [mepo-ml.zip](https://github.com/uowoolab/MEPO-ML/releases/latest/download/asset-name.zip) (can also be found in the *Release* tab); **do not clone** this repo since it does not include the trained model
3. Open a terminal in the location of the extracted code and activate the Python environment (`mamba activate mepoml` or `conda activate mepoml`)
4. Choose one of the following use cases below and run the code accordingly (example outputs can be found in the [example_output](example_output) folder)

### Use Case 1: Assigning charges for a single CIF

To predict charges for one structure (`/path/to/src/` can be absolute or relative paths):

```
python predict.py --cif /path/to/src/MOF1.cif
```

This will generate a new CIF in the original directory:

```
/path/to/src/
├── MOF1.cif
├── MOF1_mepoml.cif
└── ...
```

### Use Case 2: Writing the new CIF to a specified destination

Use the `--dst` flag to specify output location (both `/path/to/src/` and `/path/to/dst/` can be absolute or relative paths)::

```
python predict.py --cif /path/to/src/MOF2.cif --dst /path/to/dst/
```

This will generate a new CIF in the specify directory:

```
/path/to/dst/
└── MOF2_mepoml.cif
/path/to/src/
├── MOF2.cif
└── ...
```

### Use Case 3: Assigning charges for a batch of CIFs

To assign for all structures in the entire directory, use `--src` and `--dst` to specify the input and output locations (both `/path/to/src/` and `/path/to/dst/` can be absolute or relative paths):

```
python predict.py --src /path/to/src/ --dst /path/to/dst/
```

This will generate new CIFs in the target directory:

```
/path/to/dst/
├── MOF3_mepoml.cif
├── MOF4_mepoml.cif
├── MOF5_mepoml.cif
└── ...
/path/to/src/
├── MOF3.cif
├── MOF4.cif
├── MOF5.cif
└── ...
```
