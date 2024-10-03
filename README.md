# Tacotron2 

## Preparing

### Install Dependencies
1. Install Pytorch>=2.4
2. Install requirements
3. export PYTHONPATH

```shell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
pip install -r requirements.txt
export PYTHONPATH=.
```

### Environment
- OS: Windows 11, Ubuntu 22.04 LTS
- Python: 3.11
- Pytorch: 2.4.1
- GPU: A40 (48gb)

## Result

### case 0

Text: than in the same operations with ugly ones.

### case 1

Text: A further development of the Roman letter took place at Venice.

### case 2

Text: The day room was fitted with benches and settles after the manner of the tap in a public-house.

### case 3

Text: This was notably the case with the early works printed at Ulm, and in a somewhat lesser degree at Augsburg.

### case 4

Text: result in some degree of interference with the personal liberty of those involved.

### case 5

Text: We have, therefore,


## References
- Tacotron2 paper
- NVIDIA Tacotron2 Implementation
- Tacotron2 tutorial notebook