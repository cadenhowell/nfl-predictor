#!/bin/bash
python train.py passing --include_combine
python train.py passing
python train.py rushing --include_combine
python train.py rushing
python train.py receiving --include_combine
python train.py receiving