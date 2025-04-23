# Comparison of Subsampling Methods

This repository contains all materials and code for my Bachelor’s thesis, **“Comparison of Subsampling Methods”**, where we evaluate and compare different strategies for subsampling large datasets in statistical and machine‐learning contexts.

---

## 📖 Overview

Many modern applications involve datasets so large that full‐sample analysis becomes computationally expensive or infeasible. Subsampling—selecting a subset of data points—can dramatically reduce computational cost, but different methods introduce different biases and variances. In this thesis, we:

1. **Implement** several popular subsampling techniques:
   - **Uniform random subsampling**
   - **Leverage‐score based sampling**
   - **Deterministic sampling using IBOSS method**
2. **Generate synthetic test datasets** via:
   - Element‐wise (univariate) random draws
   - Row‐wise (multivariate) draws from known distributions  
3. **Compare** performance across methods in terms of estimation bias, variance, and computational efficiency.
4. **Provide** guidelines for selecting an appropriate subsampling method in practice.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher  
- [NumPy](https://numpy.org/)  
- [SciPy](https://scipy.org/)  
- [pandas](https://pandas.pydata.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- [matplotlib](https://matplotlib.org/)  

Install all dependencies with:

```bash
pip install -r requirements.txt
