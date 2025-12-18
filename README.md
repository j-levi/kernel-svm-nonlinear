# Kernel SVM — Nonlinear Classification

## Overview
This project shows how Support Vector Machines can carve out nonlinear decision boundaries by swapping inner products for kernels. It focuses on intuition around margins, support vectors, and how kernel choices bend the feature space.

## Key ideas
- A linear SVM finds the widest margin between classes; soft margins allow some mistakes when data overlap.
- In the kernelized version, you never build explicit features—you just ask a kernel function how similar two points are.
- Only support vectors matter at prediction time; everything else falls away.

## Common kernels (when to try them)
- Polynomial: captures interactions up to a chosen degree.
- RBF/Gaussian: flexible default that creates smooth, radial bumps around points.
- Sigmoid: behaves like a squashed dot product; less common but still illustrative.

## Practical tuning
- `C` trades off margin width vs. training errors; higher values push for fewer errors.
- Kernel-specific parameters (e.g., RBF bandwidth) control how wiggly the boundary gets.
- Check cross-validation accuracy and the number of support vectors to spot overfitting.

## Files
- `SVM.ipynb`: experiments with different kernels, hyperparameters, and visualizations of the resulting boundaries.
- `SVM_data.mat`: toy datasets used in the notebook.
