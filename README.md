# Entire Space Multitask Model (ESMM)

## Overview
PyTorch implementation of the paper  
[Xiao Ma, et al., Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate, SIGIR-2018](https://dl.acm.org/doi/abs/10.1145/3209978.3210104)

## Dataset
columns
- `feature1` ... `feature_n`: categorical feautre column. (Assuming all variables are categorical.)
- `click`: click label 
- `conversion`: conversion label (Always 0 if there are no clicks)

sample data
```
   feature1  feature2 feature3  click  conversion
0         1        10       cc      1           1
1         4        11        a      0           0
2         6        30        5      0           0
3        10         3       bb      1           0
4         3        33       cc      1           1
5         1         2        d      0           0
6         4         5       cd      1           0
```

## Usage
Environment construction with docker.

Start container
```
$ docker-compose up -d --build
```
Attach container
```
$ docker exec -it esmm /bin/bash
```
Run training esmm
```
$ python train.py
```