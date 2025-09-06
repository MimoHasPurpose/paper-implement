# PVNet Implementation

This is an implementation of PVNet for 6D object pose estimation using the LINEMOD dataset.

## Project Structure

```
pvnet_Implementation-main/
├── models/
│   ├── __init__.py
│   └── pvnet.py              # PVNet model implementation
├── datasets/
│   ├── __init__.py
│   └── linemod_dataset.py    # LINEMOD dataset loader
├── utils/
│   ├── __init__.py
│   ├── pnp.py               # PnP solver utilities
│   └── voting.py            # RANSAC voting utilities
├── LINEMOD/                 # LINEMOD dataset (various objects)
│   ├── cat/                 # Cat object data
│   ├── ape/                 # Ape object data
│   └── ...                  # Other objects
├── config.py               # Configuration parameters
├── training.py             # Training script
├── test.py                 # Testing script
├── createdataset.py        # Dataset creation utilities
├── createfile.py           # File parsing utilities
└── requirements.txt        # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the LINEMOD dataset is properly structured in the `LINEMOD/` directory.

## Usage

### Training
```bash
python training.py
```

### Testing
```bash
python test.py
```

### Dataset Creation
```bash
python createdataset.py
```

## Configuration

Edit `config.py` to modify:
- Number of keypoints
- Number of classes
- Camera intrinsics

## Notes

- The implementation uses relative paths for better portability
- All hardcoded Windows paths have been replaced with relative paths
- Proper package structure with `__init__.py` files
- Fixed indentation and import issues