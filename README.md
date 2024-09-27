# MEPO-ML

Jun Luo, Omar Ben Said, Peigen Xie, Marco Gibaldi, Jake Burner, Cécile Pereira & Tom K. Woo

MEPO-ML: a robust graph attention network model for rapid generation of partial atomic charges in metal-organic frameworks. *npj Comput Mater* **10**, 224 (2024). [https://doi.org/10.1038/s41524-024-01413-4](https://www.nature.com/articles/s41524-024-01413-4)

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

Currently, the code is intended to be run on CPUs only, since the model inference time is minimal compared to the cost of generating the graph and the descriptors (node features). Advanced users can install the CUDA or ROCm version of pytorch and modify codes in [assign_mepoml_charges.py](assign_mepoml_charges.py) to utilize GPU(s).

## Usage

1. Please make sure the MOF structures are converted to **CIF format and P1 symmetry** (examples inputs can be found in the [example_input](example_input) folder)
   - Please also make sure all atoms in the unit cell have occupancies of exactly 1.
2. Download and extract [MEPO-ML.zip](https://github.com/uowoolab/MEPO-ML/releases/latest/download/MEPO-ML.zip) (can also be found in the *Release* tab); **do not clone** this repo since it does not include the trained model
3. Open a terminal in the location of the extracted code and activate the Python environment (`conda activate mepoml`)
4. Choose one of the following use cases below and run the code accordingly (example outputs can be found in the [example_output](example_output) folder)


### Use Case 1: Assigning charges for a single CIF

To assign charges for one structure (`/path/to/src/` can be absolute or relative paths):

```
python assign_mepoml_charges.py --src /path/to/src/MOF1.cif
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
python assign_mepoml_charges.py --src /path/to/src/MOF2.cif --dst /path/to/dst/
```

This will generate the new CIF in the specify directory:

```
/path/to/dst/
└── MOF2_mepoml.cif
/path/to/src/
├── MOF2.cif
└── ...
```

### Use Case 3: Batch charge assignments for all CIFs in a directory

To assign charges for all structures in a directory, use `--src` and `--dst` to specify the input and output locations (both `/path/to/src/` and `/path/to/dst/` can be absolute or relative paths):

```
python assign_mepoml_charges.py --src /path/to/src/ --dst /path/to/dst/
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

**Note**: Batch assignments are also possible without `--dst`, where will store the new CIFs in `/path/to/src/`.
