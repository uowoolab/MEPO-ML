# MEPO-ML

Read our publication on *npj Computational Material*! [**DOI:** 10.1038/s41524-024-01413-4](https://www.nature.com/articles/s41524-024-01413-4)

### Citation
```
@article{luo2024mepo,
  title={MEPO-ML: a robust graph attention network model for rapid generation of partial atomic charges in metal-organic frameworks},
  author={Luo, Jun and Said, Omar Ben and Xie, Peigen and Gibaldi, Marco and Burner, Jake and Pereira, C{\'e}cile and Woo, Tom K},
  journal={npj Computational Materials},
  volume={10},
  number={1},
  pages={224},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

### Update 2.0 Highlights

This is a major update since our publication:

1. The model has been re-trained with a duplicated structures removed, and additon of ~30K experimental MOF structures.
2. During the re-train, coordination shell features are removed from the node features while bond distances are added as an edge feature. Hyperparameters of model has been re-optimized.
3. Chemical bond radii for each element pair are now used for bond table construction inplace of the Isayev's nearest neighbor algorithm. This speeds up the bond table construction significantly (since no Voronoi analysis needed) while proceducing the same bond tables.

## Python Environment

This code is tested with the following Python version and packages:
- Python **3.13.7**
- gemmi **0.7.1**
- torch **2.7.1**
- torch_geometric **2.6.1**

The next update of MEPO-ML will be pushed to `PyPI` as a package. For now, please create a virtual environment with the above requirements and follow instructructions below to use MEPO-ML.

## Usage

***Note***: Currently, the code is using CPU for inferencing. Advanced users can install the CUDA or ROCm version of pytorch and modify codes in [assign_mepoml_charges.py](assign_mepoml_charges.py) to utilize GPU(s).

1. Please make sure the MOF structures are converted to **CIF format and P1 symmetry** (examples inputs can be found in the [example_input](example_input) folder)
   - Please also make sure all atoms in the unit cell have occupancies of exactly 1.
2. Download and extract [MEPO-ML.zip](https://github.com/uowoolab/MEPO-ML/releases/latest/download/MEPO-ML.zip) (can also be found in the *Release* tab); **do not clone** this repo since it does not include the trained model
3. Open a terminal in the location of the extracted code and activate the Python environment
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
