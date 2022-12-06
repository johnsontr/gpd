# gpd - Gaussian process derivatives for marginal effects

For gpml-matlab-v3.6-2015-07-07.

# 1. Usage

## 1.1 Package functions
This section describes how to use package functions and provides a demo for general usage.

All functions rely on having already trained a Gaussian process regression model in _gpml_. All calculations in this package rely on the posterior predictive distribution of the dependent variable in the model. This package assumes MAP estimates are found by minimizing the log likelihood. In notation below, the input _hyp_ is a struct of the learned hyperparameters from training a Gaussian process regression model. For example,  

```hyp = minimize_v2(initial_hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, trainX, trainy);```

Note how _trainX_ and _trainy_ are used instead of X and y for calling `me()`. Learning hyperparameters requires normalized inputs to help with learning length scales. The normalized training inputs _X_ is _trainX_, and the normalized training outputs _y_ is _trainy_. Once model parameters are learned, prediction occurs on the un-normalized training inputs. 

### 1.1.1 _function [ mean_vec, diag_var_mat ] = me(hyp, meanfunc, covfunc, X, y, xs)_  

Core functionality of the package relies on the _me()_ function. For any test point xs, it calculates
1. the _Dx1_ vector of expected marginal effects with respect to each epxlanatory variable _k = 1, ..., D_, and  
2. the _Dx1_ vector of variances associated with the expected marginal effect.  
The _Dx1_ vector of variances in (2) is pulled from the diagonal of the variance-covariance matrix associated with (1).

Inputs:
* _hyp_
* _meanfunc_
* _covfunc_
* _X_
* _y_
* _xs_ 

Outputs: Two _Dx1_ vectors where _D_ is the number of columns in _X_. The first output is the _Dx1_ vector of expected marginal effects w.r.t. explanatory variables _k = 1, ..., D_ calculated with respect to test input _xs_. The second output is the _Dx1_ vector of diagonal entries of the variance-covariance matrix associated with the _Dx1_ output from the first position. Since diagonals are reported, the output from the function `me(hyp, meanfunc, covfunc, X, y, xs)` reports the marginal distribution of the expected marginal effect of each explanatory variable evaluated at test point _xs_.  

### 1.1.2 _function [ MEs, VARs ] = pme(hyp, meanfunc, covfunc, X, y, Xs)_  

The function calls `me(hyp, meanfunc, covfunc, X, y, Xs)` for each row _j = 1, ..., M_ in _Xs_. Since calling `me()` on a single test point _xs_ produces two _Dx1_ vectors, calling `pme()` on _MxD_ test data _Xs_ produces two _DxM_ vectors. When _Xs_ is omitted from the function call, then the function assumes that marginal effect calculations are made with respect to the training inputs (i.e., _Xs = X_).   

Inputs:
* _hyp_
* _meanfunc_
* _covfunc_
* _X_
* _y_
* _Xs_ (optional)  

Outputs: Two _DxM_ vectors. The jth column of the first output is the _Dx1_ vector of epxected marginal effects associated with the jth test point _Xs(j,:)_. The jth column of the second output is the _Dx1_ vector of variances associated with the expected marginal effect of each of the _D_ explanatory variables evaluated at test point _Xs(j,:)_.  


### 1.1.3 _function [gmm_mean, gmm_mean_var, cred95] = ame(hyp, meanfunc, covfunc, X, y, Xs)_  

Calls `pme(hyp, meanfunc, covfunc, X, y, Xs)` to generate summary statistics across the test inputs using general method of moments. When _Xs_ is omitted from the function call, then the function assumes that marginal effect calculations are made with respect to the training inputs (i.e., _Xs = X_).  

Inputs:
* _hyp_
* _meanfunc_
* _covfunc_
* _X_
* _y_
* _Xs_ (optional)  

Outputs: A _Dx1_ vector where the kth entry is the expected marginal effect of the kth explanatory variable from a `pme()` call averaged across the _M_ test inputs in _Xs_ from the `pme()` call (i.e., a mean of marginal effects), a _Dx1_ vector of the sample variance corresponding to the first output (i.e., the variance of marginal effects), and a _Dx2_ matrix of 95% credible intervals calculated for each dimension. When Xs is omitted from the input, calculations are made on the training sample so that _Xs = X_.

### 1.1.4 _function plt = plotme(d, hyp, meanfunc, covfunc, X, y, Xs, ~)_  

The function `plotme()` is the main plotting function of the package. Other plotting functions such as `gridme()` (see below) depend on it. Returns a plot object that is already labeled appropriately. When the final two inputs are omitted, then calculations are made with respect to the training sample so that _Xs = X_. When _Xs_ is specified but the final function input is omitted, calculations are made with respect to predictions over test inputs _Xs_. When any value is passed to the final (eighth) function input, then `plotme()` omits some information for the plot so it can be customized for plotting interactions. The last function input is used whenever `gridme()` is passed inputs indicating interactions are of interest.

Inputs:
* _d_
* _hyp_
* _meanfunc_
* _covfunc_
* _X_
* _y_
* _Xs_ (optional)  
* _~_ (optional)

Output: A plot object.


### 1.1.5 _function [ plt, gridX ] = gridme(d, numsteps, hyp, meanfunc, covfunc, X, y, interaction_indices)_  

The function `gridme()` automates some of the prediction process. The function `gridme()` calls `plotme()` and generates gridded data for the dth dimension with other explanatory variables held at their mean. The grid will have _numpsteps_ points. When _interaction_indices_ is specified, then all dimensions in the vector _interaction_indices = [k1, k2, ...]_ will be gridded when making predictions.

Inputs:
* _d_
* _hyp_
* _meanfunc_
* _covfunc_
* _X_
* _y_
* _Xs_ (optional)  
* _~_ (optional)

Outputs: A plot object.


## 1.2 Package demo

# 2. Simulations

The remainder produces plots which verify that the method is working correctly.

## 2.1 Univariate functions

### 2.1.2 Linear, quadratic, and cubic expansions without fixed or random effects

Gaussian process regression models applied to functions with a single input are equivalent when specified with either covSEiso or covSEard.

| Linear | Quadratic | Cubic |
:---:|:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_linear_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_quadratic_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_cubic_iso.png)

### 2.1.3 Linear, quadratic, and cubic expansions with fixed effects

Encode fixed effects as one-hot.

| Linear+FE | Quadratic+FE | Cubic+FE |
:---:|:---:|:---:
![]() | ![]() | ![]()

## 2.2 Bivariate functions with iid N(0,1) explanatory variables and no fixed or random effects

### covSEiso

#### 2.2.1 Bivariate linear function and no fixed or random effects
| X1 | X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x1_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x2_iso.png)

#### 2.2.2 Bivariate linear function with interactions and no fixed or random effects
| X1 | X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x1_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x2_iso.png)

### covSEard

#### 2.2.3 Bivariate linear function and no fixed or random effects
| X1 | X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x1_ard.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x2_ard.png)

#### 2.2.4 Bivariate linear function with interactions and no fixed or random effects
| X1 | X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x1_ard.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x2_ard.png)

## 2.3 Bivariate functions with jointly normal explanatory variables and no fixed or random effects

### covSEiso

#### 2.3.1 Bivariate linear function and no fixed or random effects

#### 2.3.2 Bivariate linear function with interactions and no fixed or random effects

### covSEard

#### 2.3.3 Bivariate linear function and no fixed or random effects

#### 2.3.4 Bivarite linear function with interactions and no fixed or random effects

