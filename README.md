# Assessing AlphaFold3 Predictions for Protein-Ligand Affinity via Sfcnn


## Overview

This project systematically evaluates the reliability of AlphaFold3 (AF3)-predicted protein structures for protein-ligand affinity (PLA) prediction tasks. The study reproduces the Sfcnn model, a 3D convolutional neural network (CNN) for PLA prediction, using PyTorch and assesses AF3-derived protein structures compared to experimentally determined structures.

## Data Source
Due to the size of the training data, it is not included in the final submission,
please download it from **https://send.deemos.com/download/d9f9dabbfe31319d/#AgG7mUgd3kPtOyl4TR4Xvw** through code: **m9QsPj**.

**Please make sure file storage follows the project tree below.** 

## Project Structure

```
CS177_pj/
├── README.md
├── data/                              # Dataset files
│   ├── core_affinity_2016.csv
│   ├── core_affinity_final.csv
│   ├── sfcnn_out.csv
│   ├── chai_hdf5/                     # HDF5 data files
│   ├── chai_results_cif/              # CIF structure files
│   ├── chai_results_pdb/              # PDB structure files
│   ├── chai_results_zip/              # Compressed results
│   ├── coreset/                       # CASF-2016 core set
│   ├── PDBbind_v2019_plain_text_index/
│   ├── refined-set/                   # PDBbind v2019 refined set
│   ├── test_hdf5/                     # Test data in HDF5 format
│   └── train_hdf5/                    # Training data in HDF5 format
└── src/
    ├── sfcnn/                         # Main Sfcnn implementation
    │   ├── src/
    │   │   ├── data.py               # Data preprocessing and featurization
    │   │   ├── train.py              # Model training with k-fold CV
    │   │   ├── predict.py            # Prediction interface
    │   │   ├── eval_casf2016.py      # CASF-2016 evaluation
    │   │   ├── readloss.py           # Training metrics visualization
    │   │   └── train_results/        # Training outputs and models
    │   ├── outputs/                   # Evaluation results
    │   └── targets/                   # Target files
    ├── AF3_eval/                      # AlphaFold3 evaluation
    │   ├── src/
    │   │   ├── data.py               # AF3-specific data processing
    │   │   ├── eval.py               # AF3 structure evaluation
    │   │   ├── parse.py              # Results parsing and analysis
    │   │   ├── predict.py            # AF3-aware prediction
    │   │   └── utils/                # Utility functions
    │   ├── model/                     # Pre-trained models
    │   └── outputs/                   # AF3 evaluation results


```

## Key Components

### 1. Sfcnn Model Reproduction (`src/sfcnn/`)

The core implementation reproduces the Sfcnn neural network for protein-ligand affinity prediction:

- **data.py**: Used to generate the h5 training/testing data, notice the output is about 40.1 GiB
- **train.py**: Main training file, run directly to initiate K-fold training, results will be stored in `src/sfcnn/src/train_results`
- **readloss.py**: Visualization of the training process, please
run it after the training process
- **predict.py**: Middleware used for model loading and prediction,
which can also be runned for single protein prediction
- **eval_casf2016.py**: Visualization and metrics comparision for 
model performance on casf2016 test set


### 2. AlphaFold3 Evaluation (`src/AF3_eval/`)

Systematic assessment of AF3-predicted structures:

- **utils**: Intermidiate tools for .cif extraction etc
- **data.py**: Used to generate the h5 testing data, directly with .cif
- **predict.py**: Exact middleware in sfcnn
- **eval.py**: Result metrics generation using AF3 predicted structures
- **parse.py**: Final result visualization and metrics comparision


## Requirements

### Python Dependencies

```txt
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
h5py>=3.1.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
tqdm>=4.62.0
openbabel>=3.1.0
biopython>=1.79
```

### System Requirements

- **GPU**: CUDA-compatible GPU recommended for training
- **Memory**: Minimum 16GB RAM (32GB+ recommended for full dataset)
- **Storage**: ~50GB for complete dataset


## Usage

### Training the Sfcnn Model

```bash
python src/sfcnn/src/train.py --batch 32 --dropout 0.15 --lr 0.00068 --k_folds 7
```

### Making Predictions

```bash
python src/sfcnn/src/predict.py \
    --protein /path/to/protein.pdb \
    --ligand /path/to/ligand.mol2 \
    --weights /path/to/model_weights.pt \
    --output predictions.txt
```

### Evaluating on CASF-2016

```bash
python src/sfcnn/src/eval_casf2016.py
```

### AlphaFold3 Structure Evaluation

```bash
python src/AF3_eval/src/eval.py
```

### Visualizing Training Results

```bash
python src/sfcnn/src/readloss.py --mode summary  # Cross-validation summary
python src/sfcnn/src/readloss.py --mode fold --fold 1  # Specific fold results
```



