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

### Train

1. data preprocessing
2. run train script

```shell
python scripts/preprocessing.py -cn ljspeech
python scripts/train.py -cn ljspeech
```

### Environment
- OS: Windows 11, Ubuntu 22.04 LTS
- Python: 3.11
- Pytorch: 2.4.1
- GPU: A40 (48gb)

## Result

### case 0

Text: than in the same operations with ugly ones.

![case0](https://raw.githubusercontent.com/crlotwhite/Tacotron2/refs/heads/master/docs/case0.png)

### case 1

Text: A further development of the Roman letter took place at Venice.

![case1](https://raw.githubusercontent.com/crlotwhite/Tacotron2/refs/heads/master/docs/case1.png)

### case 2

Text: The day room was fitted with benches and settles after the manner of the tap in a public-house.

![case2](https://raw.githubusercontent.com/crlotwhite/Tacotron2/refs/heads/master/docs/case2.png)

### case 3

Text: This was notably the case with the early works printed at Ulm, and in a somewhat lesser degree at Augsburg.

![case3](https://raw.githubusercontent.com/crlotwhite/Tacotron2/refs/heads/master/docs/case3.png)

### case 4

Text: result in some degree of interference with the personal liberty of those involved.

![case4](https://raw.githubusercontent.com/crlotwhite/Tacotron2/refs/heads/master/docs/case4.png)

### case 5

Text: We have, therefore,

![case5](https://raw.githubusercontent.com/crlotwhite/Tacotron2/refs/heads/master/docs/case5.png)


## References
- Tacotron2 paper
- NVIDIA Tacotron2 Implementation
- Tacotron2 tutorial notebook