# MEPO-ML
A graph attention network based model for predicting atomic partial charges in metal-organic frameworks.

### Notes

This project is still **WORK IN PROGRESS**. Currently only includes minimum codes to use the model to generate the charges (i.e. inference only), which is ready for production. Codes and files for training new models or reproducing our results will be added in the future.

## Prerequisites

This code is tested with the following Python version and packages:
- Python **3.11.7**
- pymatgen **2024.2.8**
- pytorch **2.1.2**
- pytorch_geometric **2.4.0**

We recommand using `conda` to create a separate environment to use MEPO-ML; for minimal installations, [install miniconda](https://docs.anaconda.com/free/miniconda/). Once installed, run the following commands (one line at a time) to create the MEPO-ML environment:

```
conda create --name mepoml python=3.11 
conda activate mepoml
pip install pymatgen=2024.2.8 torch=2.1 torch_geometric=2.4
```

Currently, the code is intended to be run on CPUs only, since the model inference time is minimal compared to the cost of generating the graph and the descriptors (node features). Advanced users can install the CUDA or ROCm version of pytorch and modify codes in [predict.py](predict.py) to utilize GPU(s).

## Usage

1. Please make sure the MOF structures are converted to **CIF format and P1 symmetry** (examples inputs can be found in the [example_input](example_input) folder)
2. Download and extract [mepo-ml.zip](https://github.com/uowoolab/MEPO-ML/releases/latest/download/asset-name.zip) (can also be found in the *Release* tab); **do not clone** this repo since it does not include the trained model
3. Open a terminal in the location of the extracted code and activate the Python environment (`conda activate mepoml`)
4. Choose one of the following use cases below and run the code accordingly (example outputs can be found in the [example_output](example_output) folder)


### Use Case 1: Assigning charges for a single CIF

To assign charges for one structure (`/path/to/src/` can be absolute or relative paths):

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

This will generate the new CIF in the specify directory:

```
/path/to/dst/
└── MOF2_mepoml.cif
/path/to/src/
├── MOF2.cif
└── ...
```

### Use Case 3: Assigning charges in batches

To assign charges for all structures in a directory, use `--src` and `--dst` to specify the input and output locations (both `/path/to/src/` and `/path/to/dst/` can be absolute or relative paths):

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
