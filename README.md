# gpd - Gaussian process derivatives for marginal effects

*Dependencies*: 
* _GNU Octave_ versions 3.2.x and higher **or** _MATLAB_ versions 7.x and higher
* _gpml-matlab-v3.6-2015-07-07_

This package implements a routine to estimate Gaussian process regression model derivatives found in notes by a former Ph.D. student Andrew McHutchon that was associated with [Carl Edward Rasmussen's machine learning group](https://mlg.eng.cam.ac.uk/carl/). Andrew McHutchon's Cambridge website is no longer operational, so these notes were accessed through [the most recent archived version available from the Wayback Machine](https://web.archive.org/web/20210225174148/https://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf). Andrew McHutchon's notes are dated April 17, 2013, and the most recent Wayback Machine archive is from February 25, 2021.


This method has other applications in engineering and the natural sciences, all of which cite Andrew McHutchon's unpublished working paper. Citations of Andrew McHutchon's "Differentiating Gaussian Processes" working paper can be found on [SemanticScholar](https://www.semanticscholar.org/paper/Differentiating-Gaussian-Processes-McHutchon/3ad0725b8dd4eb32ca2a27f25d522741293a5252).

This method has been implemented in political science in a [published paper](https://jbduckmayr.com/publication/gpirt/) as well as a [working paper](https://jbduckmayr.com/working-papers/inference-in-gp-models/) by [JBrandon Duck-Mayr](https://jbduckmayr.com/).

1. [GPIRT: A Gaussian Process Model for Item Response Theory](https://proceedings.mlr.press/v124/duck-mayr20a.html)
2. [Inference in Gaussian Process Models for Political Science](https://jbduckmayr.com/working-papers/inference-in-gp-models/inference-in-gp-models.pdf)

JB has implemented routines from these two papers with _R_ packages [gpirt](https://github.com/duckmayr/gpirt) and [gpmss](https://github.com/duckmayr/gpmss).

[Herb Susmann](https://herbsusmann.com/) has some applications of derivatives of Gaussian processes on his blog.
1. [July 7, 2020 - Derivatives of a Gaussian Process](https://herbsusmann.com/2020/07/06/gaussian-process-derivatives/)
2. [December 1, 2020 - Conditioning on Gaussian Process Derivative Observations](https://herbsusmann.com/2020/12/01/conditioning-on-gaussian-process-derivative-observations/)
3. [December 11, 2020 - Derivative Gaussian Processes in Stan](https://herbsusmann.com/2020/12/11/derivative-gaussian-processes-in-stan/)


# 1. Using the _gpd_ package

This section describes how to use package functions and provides a demo for general usage.

## 1.1 Package functions

There are five functions in the package _gpd_.
1. _[ mean_vec, diag_var_mat ] = me(hyp, meanfunc, covfunc, X, y, xs)_
2. _[ MEs, VARs ] = pme(hyp, meanfunc, covfunc, X, y, Xs)_  
3. _[gmm_mean, gmm_mean_var, cred95] = ame(hyp, meanfunc, covfunc, X, y, Xs)_ 
4. _plt = plotme(d, hyp, meanfunc, covfunc, X, y, Xs, ~)_ 
5. _[ plt, gridX ] = gridme(d, numsteps, hyp, meanfunc, covfunc, X, y, interaction_indices)_

All functions rely on having already trained a Gaussian process regression model in _gpml_. All calculations in this package rely on the posterior predictive distribution of the dependent variable in the model. This package assumes MAP estimates are found by minimizing the log likelihood. In notation below, the input _hyp_ is a struct of the learned hyperparameters from training a Gaussian process regression model. For example,  

```hyp = minimize_v2(initial_hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, trainX, trainy);```

Note how _trainX_ and _trainy_ are used instead of X and y for calling `me()`. Learning hyperparameters requires normalized inputs to help with learning length scales. The normalized training inputs _X_ is _trainX_, and the normalized training outputs _y_ is _trainy_. Once model parameters are learned, this package requires the _non-normalized_** training inputs _(X,y)_.

### 1.1.1 _[ mean_vec, diag_var_mat ] = me(hyp, meanfunc, covfunc, X, y, xs)_  

Core functionality of the package relies on the _me()_ function. For any test point xs, it calculates:  

1. the _Dx1_ vector of expected marginal effects with respect to each epxlanatory variable _k = 1, ..., D_, and  
2. the _Dx1_ vector of variances associated with the expected marginal effect.  

The _Dx1_ vector of variances in (2) is pulled from the diagonal of the variance-covariance matrix associated with (1).

**Inputs**:
* _hyp_ - Model hyperparameters learned from the parent gpml model
* _meanfunc_ - The meanfunc of the corresponding gpml model (only _{@meanZero}_ is supported at this time)
* _covfunc_ - The covfunc of the parent gpml model (only _{@covSEiso}_ and _{@covSEard}_ are supported at this time)
* _X_ - The _NxD_ non-normalized training inputs used when training the parent gpml model
* _y_ - The _Nx1_ non-normalized training outputs used when training the parent gpml model
* _xs_  - A _1xD_ non-normalized test point

**Outputs**: Two _Dx1_ vectors where _D_ is the number of columns in _X_. The first output is the _Dx1_ vector of expected marginal effects w.r.t. explanatory variables _k = 1, ..., D_ calculated with respect to test input _xs_. The second output is the _Dx1_ vector of diagonal entries of the variance-covariance matrix associated with the _Dx1_ output from the first position. Since diagonals are reported, the output from the function `me(hyp, meanfunc, covfunc, X, y, xs)` reports the marginal distribution of the expected marginal effect of each explanatory variable evaluated at test point _xs_.  

### 1.1.2 _[ MEs, VARs ] = pme(hyp, meanfunc, covfunc, X, y, Xs)_  

The function calls `me(hyp, meanfunc, covfunc, X, y, Xs)` for each row _j = 1, ..., M_ in _Xs_. Since calling `me()` on a single test point _xs_ produces two _Dx1_ vectors, calling `pme()` on _MxD_ test data _Xs_ produces two _DxM_ vectors. When _Xs_ is omitted from the function call, then the function assumes that marginal effect calculations are made with respect to the training inputs (i.e., _Xs = X_).   

**Inputs**:
* _hyp_ - Model hyperparameters learned from the parent gpml model
* _meanfunc_ - The meanfunc of the corresponding gpml model (only _{@meanZero}_ is supported at this time)
* _covfunc_ - The covfunc of the parent gpml model (only _{@covSEiso}_ and _{@covSEard}_ are supported at this time)
* _X_ - The _NxD_ non-normalized training inputs used when training the parent gpml model
* _y_ - The _Nx1_ non-normalized training outputs used when training the parent gpml model
* _Xs_ (optional) - A _MxD_ non-normalized matrix of test points

**Outputs**: Two _DxM_ vectors. The jth column of the first output is the _Dx1_ vector of epxected marginal effects associated with the jth test point _Xs(j,:)_. The jth column of the second output is the _Dx1_ vector of variances associated with the expected marginal effect of each of the _D_ explanatory variables evaluated at test point _Xs(j,:)_.  

### 1.1.3 _[gmm_mean, gmm_mean_var, cred95] = ame(hyp, meanfunc, covfunc, X, y, Xs)_  

Calls `pme(hyp, meanfunc, covfunc, X, y, Xs)` to generate summary statistics across the test inputs using general method of moments. When _Xs_ is omitted from the function call, then the function assumes that marginal effect calculations are made with respect to the training inputs (i.e., _Xs = X_).  

**Inputs**:
* _hyp_ - Model hyperparameters learned from the parent gpml model
* _meanfunc_ - The meanfunc of the corresponding gpml model (only _{@meanZero}_ is supported at this time)
* _covfunc_ - The covfunc of the parent gpml model (only _{@covSEiso}_ and _{@covSEard}_ are supported at this time)
* _X_ - The _NxD_ non-normalized training inputs used when training the parent gpml model
* _y_ - The _Nx1_ non-normalized training outputs used when training the parent gpml model
* _Xs_ (optional) - A _MxD_ non-normalized matrix of test points

**Outputs**: A _Dx1_ vector where the kth entry is the expected marginal effect of the kth explanatory variable from a `pme()` call averaged across the _M_ test inputs in _Xs_ from the `pme()` call (i.e., a mean of marginal effects), a _Dx1_ vector of the sample variance corresponding to the first output (i.e., the variance of marginal effects), and a _Dx2_ matrix of 95% credible intervals calculated for each dimension. When Xs is omitted from the input, calculations are made on the training sample so that _Xs = X_.

### 1.1.4 _plt = plotme(d, hyp, meanfunc, covfunc, X, y, Xs, ~)_  

The function `plotme()` is the main plotting function of the package. Other plotting functions such as `gridme()` (see below) depend on it. Returns a plot object that is already labeled appropriately. When the final two inputs are omitted, then calculations are made with respect to the training sample so that _Xs = X_. When _Xs_ is specified but the final function input is omitted, calculations are made with respect to predictions over test inputs _Xs_. When any value is passed to the final (eighth) function input, then `plotme()` omits some information for the plot so it can be customized for plotting interactions. The last function input is used whenever `gridme()` is passed inputs indicating interactions are of interest.

**Inputs**:
* _d_ - The explanatory variable _d_ in _{ 1, ..., D }_ for which plots will be made
* _hyp_ - Model hyperparameters learned from the parent gpml model
* _meanfunc_ - The meanfunc of the corresponding gpml model (only _{@meanZero}_ is supported at this time)
* _covfunc_ - The covfunc of the parent gpml model (only _{@covSEiso}_ and _{@covSEard}_ are supported at this time)
* _X_ - The _NxD_ non-normalized training inputs used when training the parent gpml model
* _y_ - The _Nx1_ non-normalized training outputs used when training the parent gpml model
* _Xs_ (optional) - A _MxD_ non-normalized matrix of test points
* _~_ (optional) - A placeholder for the function `gridme()` that denotes whether to omit some plot information; any non-empty input in the eigth position activates this function feature

**Outputs**: A plot object.

### 1.1.5 _[ plt, gridX ] = gridme(d, numsteps, hyp, meanfunc, covfunc, X, y, interaction_indices)_  

The function `gridme()` automates some of the prediction process. The function `gridme()` calls `plotme()` and generates gridded data for the dth dimension with other explanatory variables held at their mean. The grid will have _numpsteps_ points. When _interaction_indices_ is specified, then all dimensions in the vector _interaction_indices = [k1, k2, ...]_ will be gridded when making predictions.

**Inputs**:
* _d_ - The explanatory variable _d_ in _{ 1, ..., D }_ for which plots will be made. Gridding on dimension _d_ creates _numsteps_ observations over _min(X(:,d)) - 2*sqrt(var(X(:,d)))_ and _max(X(:,d)) + 2*sqrt(var(X(:,d)))_ with other explanatory variables held at their mean.
* _numsteps_ - 
* _hyp_ - Model hyperparameters learned from the parent gpml model
* _meanfunc_ - The meanfunc of the corresponding gpml model (only _{@meanZero}_ is supported at this time)
* _covfunc_ - The covfunc of the parent gpml model (only _{@covSEiso}_ and _{@covSEard}_ are supported at this time)
* _X_ - The _NxD_ non-normalized training inputs used when training the parent gpml model
* _y_ - The _Nx1_ non-normalized training outputs used when training the parent gpml model
* _interaction_indices_ (optional) - A vector with length between 2 and _D_ with unique integer entries between 1 and D that specify which dimensions of the explanatory variables are to be gridded. Each explanatory variable is gridded so that _numsteps_ observations are made over _min(X(:,d)) - 2*sqrt(var(X(:,d)))_ and _max(X(:,d)) + 2*sqrt(var(X(:,d)))_. All dimensions not specified in _interaction_indices_ are held at their mean.

**Outputs**: A plot object and the gridded data used to generate the plot.

## 1.2 Package demo

In this subsection, I demonstrate general usage based on a simple test case. I create data with a univariate data generating process having known properties and demonstrate each function's usage.



# 2. Simulations

This section provides plots from `gridme()` which demonstrate the method more than the _gpd_ package itself. In particular, the plots for data generating processes with interactions between explanatory variables show that the method demonstrates tremendous flexibility when fitting marginal effects that may vary w.r.t. to other explanatory variables. 

## 2.1 Univariate functions

### 2.1.2 Linear, quadratic, and cubic expansions

Gaussian process regression models applied to functions with a single input are equivalent when specified with either covSEiso or covSEard.

| Linear | Quadratic | Cubic |
:---:|:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_linear_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_quadratic_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/univariate_cubic_iso.png)

## 2.2 Bivariate functions with independent normal covariates

#### 2.2.1 Linear

| covSEiso - X1 | covSEiso - X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x1_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x2_iso.png)

| covSEard - X1 | covSEard - X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x1_ard.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_x2_ard.png)

#### 2.2.2 Linear with interactions

| covSEiso - X1 | covSEiso - X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x1_iso.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x2_iso.png)

| covSEard - X1 | covSEard - X2 |
:---:|:---:
![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x1_ard.png) | ![](https://github.com/johnsontr/gpd/blob/main/simulations/results/bivariate_linear_interaction_x2_ard.png)

## 2.3 Bivariate functions with jointly normal covariates

#### 2.3.1 Linear

#### 2.3.2 Linear with interactions

