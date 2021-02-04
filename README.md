# Introduction

Code accompanying the paper "Empowering Adaptive Early-Exit Inference with Latency Awareness".

## Dependencies

+ Python 3.7.6
+ NumPy 1.18.1
+ SciPy 1.4.1
  
Note that although we implement the early-exit models using PyTorch 1.4.0, our proposed method can be implemented and evaluated without the dependency of PyTorch. This is because the required confidences and 0-1 losses are all contained in `experiment_data` folder. As an example, for the experiments on MSDNet, the confidences of 50000 test images' outputs are recorded by `experiment_data/MSD_test_confidence.npy`, which is a 5-by-50000 NumPy array with the (i,j)-th entry corresponding to the confidence of j-th image's output at i-th exit point; the 0-1 losses of 50000 test images are recorded by `experiment_data/MSD_test_correct.npy`, which is also a 5-by-50000 NumPy array with the (i,j)-th entry indicating whether j-th image's inference result given by i-th exit point is correct.

## Usage

To run the experiments on B-AlexNet:

```bash
#!/bin/bash
python Alex_ippp_val.py --exp
```

To run the experiments on S-ResNet-18:

```bash
#!/bin/bash
python Scan_ippp_val.py --exp
```

To run the experiments on MSDNet:

```bash
#!/bin/bash
python MSD_ippp_val.py --exp
```

To run the experiments on DSA-19:

```bash
#!/bin/bash
python DSA_ippp_val.py --exp
```

To run the experiments on Google-30:

```bash
#!/bin/bash
python Speech_ippp_val.py --exp
```

The experimental scripts come with several options, which can be listed with the `--help` flag.
