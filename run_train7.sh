#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONHASHSEED=42
python train7.py
