
# SIFAR-Hiera: Super Image for Action Recognition using Hiera

This repository focuses on finetuning **Hiera (Hierarchical Vision Transformer – Small and Base variants)** on video classification datasets using **SIFAR-style Super Image representations**.

---

##  Overview

**SIFAR (Super Image for Action Recognition)** enables video understanding using standard image models by converting a sequence of frames into a **single Super Image**.

Instead of temporal modeling, frames are rearranged spatially into a grid (e.g., 4×4 or 3×3), allowing efficient processing using image-based backbones.

In this work, we utilize **Hiera (Hierarchical Vision Transformer)** as the backbone, which is designed for efficient multi-scale feature learning and better spatial hierarchy modeling.

---

##  Architecture

**Pipeline Overview:**

```
Video Frames
     │
     ▼
Frame Sampling
     │
     ▼
Super Image Construction (SIFAR)
     │
     ▼
Hiera Backbone (Small / Base)
     │
     ▼
Classification Head
     │
     ▼
Action Prediction
```

---

##  Super Image Configurations

Two Super Image settings are used:

| Configuration | Grid Size | Frames (Duration) | super_img_rows |
| ------------- | --------- | ----------------- | -------------- |
| **4×4 SIFAR** | 4 × 4     | 16 frames         | 4              |
| **3×3 SIFAR** | 3 × 3     | 8 frames          | 3              |

---

##  Quick Start

### 1. Environment Setup

```bash
conda env create --file env.yaml
conda activate sifar_msn
```

---

##  Dataset

Currently supported:

* **Kinetics 400, SSv2, UCF 101 and HMDB51**

Ensure annotation format:

```
/path/to/video.mp4 label start_frame end_frame
```

---

##  Training

###  1. Hiera Base – 4×4 Super Image (Duration = 16)

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=28528 main.py \
  --data_dir /path/to/hmdb51 \
  --use_pyav --dataset hmdb51 \
  --opt adamw --lr 1e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model hiera_base \
  --output_dir /path/to/output \
  --weight-decay 0.01 --clip-grad 2.0 \
  --class_numbers 51
```

---

###  2. Hiera Small – 4×4 Super Image (Duration = 16)

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=28527 main.py \
  --data_dir /path/to/hmdb51 \
  --use_pyav --dataset hmdb51 \
  --opt adamw --lr 1e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model hiera_small \
  --output_dir /path/to/output \
  --weight-decay 0.01 --clip-grad 2.0 \
  --class_numbers 51
```

---

###  3. Hiera Base – 3×3 Super Image (Duration = 8)

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=28527 main.py \
  --data_dir /path/to/hmdb51 \
  --use_pyav --dataset hmdb51 \
  --opt adamw --lr 1e-4 --epochs 30 --sched cosine \
  --duration 8 --batch-size 4 --super_img_rows 3 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model hiera_base \
  --output_dir /path/to/output \
  --weight-decay 0.01 --clip-grad 2.0 \
  --class_numbers 51
```

---

## Key Notes

* **Backbone:** Hiera (Small & Base)
* **Input Representation:** Super Image (SIFAR)
* **Dataset:** HMDB51
* **Training Strategy:** Standard finetuning with pretrained weights
* **Optimizer:** AdamW + Cosine Scheduler

---

## Acknowledgements

This work builds upon:

* **SIFAR: Super Image for Action Recognition**
* **Hiera: Hierarchical Vision Transformer**

---

##  Contact

**Author:** Sudipta Sarkar
**Date:** 10 May 2025
**Email:** [sudiptasarkar3600@gmail.com](mailto:sudiptasarkar3600@gmail.com)

---
