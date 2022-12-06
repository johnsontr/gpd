# gpd - Gaussian process derivatives for marginal effects

For gpml-matlab-v3.6-2015-07-07.

# 1. Usage

## 1.1 Package functions
This section describes how to use package functions and provides a demo for general usage.

### 1.1.1 me

### 1.1.2 pme

### 1.1.3 ame

### 1.1.4 plotme

### 1.1.5 gridme

## 1.2 Package demo

# 2. Simulations

The remainder produces plots which verify that the method is working correctly.

## 2.1 Univariate functions

### 2.1.2 covSEiso = covSEard

| Linear | Quadratic | Cubic |
:---:|:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_linear_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_quadratic_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_cubic_iso.png)

## 2.2 Bivariate functions

### 2.2.1 covSEiso

#### 2.2.1.1 Bivariate linear function
| X1 | X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x1_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x2_iso.png)

#### 2.2.1.2 Bivariate linear function with interactions
| X1 | X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x1_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x2_iso.png)


### 2.2.2 covSEard

#### 2.2.2.2 Bivariate linear function
| X1 | X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x1_ard.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x2_ard.png)

#### 2.2.2.2 Bivariate linear function with interactions
| X1 | X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x1_ard.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x2_ard.png)

