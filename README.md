# Kernel SVM — Nonlinear Classification (Mathematical Notes)

## Overview
This project demonstrates Support Vector Machines (SVMs) with kernels to handle nonlinear decision boundaries. The README explains primal/dual formulations, the kernel trick, and common kernels.

## Linear SVM (Primal)
For linearly separable data {(x_i, y_i)} with y_i ∈ {+1, -1}, the hard-margin SVM solves:
$$
\min_{w,b}\ \frac{1}{2} \|w\|^2 \quad \text{s.t. } y_i (w^T x_i + b) \ge 1,\ \forall i.
$$
Soft-margin adds slack variables ξ_i ≥ 0:
$$
\min_{w,b,\xi} \ \frac{1}{2}\|w\|^2 + C \sum_i \xi_i \quad \text{s.t. } y_i(w^T x_i + b) \ge 1 - \xi_i.
$$

## Dual Formulation and Kernel Trick
Form the Lagrangian and derive the dual problem:
$$
\max_{\alpha} \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j
\\
\text{s.t. } 0 \le \alpha_i \le C,\ \sum_i \alpha_i y_i = 0.
$$
Replace inner products with a kernel \(k(x_i,x_j)=\phi(x_i)^T\phi(x_j)\) to get the kernelized dual:
$$
\max_{\alpha} \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j k(x_i,x_j).
$$
Decision function for a test point x:
$$
f(x) = \sum_i \alpha_i y_i k(x_i,x) + b, \quad \hat y = \mathrm{sign}(f(x)).
$$
Only support vectors (α_i > 0) contribute to the sum.

## Typical Kernels
- Polynomial: \(k(x,y) = (x^T y + c)^d\).
- Radial Basis Function (RBF / Gaussian):
$$
k(x,y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right).
$$
- Sigmoid: \(k(x,y)=\tanh(κ x^T y + θ)\).

## Solvers and Practicalities
- The dual quadratic program (QP) can be solved by SMO (Sequential Minimal Optimization) or convex solvers.
- Kernel matrix K must be positive semi-definite.
- Regularization parameter C balances margin width vs. misclassification.

## References
- Cortes & Vapnik, "Support-vector networks", 1995.
- Scholkopf & Smola, Learning with Kernels.

## Files
See SVM.ipynb and datasets in this folder (SVM_data.mat).
