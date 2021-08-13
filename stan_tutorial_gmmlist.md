---
title: "GMM Estimation Using Stan - Tutorial and List of Models"
author: "Nicolas Kuehn and Peter Stafford"
date: "12 August, 2021"
output:
  html_document:
    keep_md: true
    toc: true
    toc_depth: 2
    number_sections: true
    highlight: tango
  pdf_document: default
bibliography: references.bib
---




# Introduction

This is a list of Stan models for GMM development.

# Getting Started

This tutorial uses **Stan** version 2.24.1.

```
## [1] "/Users/nico/GROUNDMOTION/SOFTWARE/cmdstan-2.27.0"
```

```
## [1] "2.27.0"
```


# A simple Stan Program for Ground-Motion Model Regression

A Stan program is made up of blocks, like a `data {}`, `parameters {}` and a `model {}` block.
These are used to declare the data, the parameters to be estimated, and a generative model for the data.
A declaration of a variable will look like `real a;` to declare a variable `a` that is a real, or `vector[N] Y;` to declare a vector of length `N`.
Stan is typed, so there is a difference between a declaration `real a;` or `int a;`.
Constraints can be declared as `real<lower=L,upper=U> a;`, which means that `a` can take only values `L <= a <= U`.
Each line in a stan program has to end in `;`.

In the Stan program below, we first declare the number of records `N` and the number of events `NEQ` as integer values.
We then declare the target and predictor variables as vectors of length `N`.
Alternatively, they could also be declared as an array, via `real M[N]`.
We also declare an integer array `int idx_eq[N]` which stores the event indices as numbers between `1` and `NEQ`.

Next we have a `transformed data {}` block.
This block is optional, and can be used to define global variables that are used throughout the program.

In the `parameters {}` block the coefficients `theta`, the standard deviations `phi` and `tau` and the event terms `deltaB` are declared.
Since standard deviations need to be positive, they are declared as `real<lower=0> tau` and `real<lower=0> phi`.

The `model {}` block contains the generative model, which includes the functional form for our GMPE, but also defines the prior distributions for the parameters to be estimated.
There is a loop over all records, and inside the loop we caclulate the median prediction for each record (including the event term).
The data is assumed to be normally-distributed with mean equal to the event term corrected median prediction and standard deviation `phi`.
The loop shows the declaration of a local variable `mu`, which is local to the `for` loop.
Since Stan estimates parameters via Bayesian inference, we need to specify prior distributions for the parameters, which are the `theta1 ~ normal(0,10)` statements - in this case, the prior distribution for `theta1` is a normal distribution with mean zero and standard deviation 10.
If no prior distributions are specified, Stan will assume an improper uniform prior over the values for which the parameter is declared.
Specification of prior distributions is an important and often-discussed topic - we recommend checking out the prior recommendation wiki for some guidelines (<https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations>).
The standard deviations are given half-cauchy distributions (since they are constrained to be positive), which is the default recommendation in Stan.
In our experience, these work well for $\phi$, $\tau$ and $\phi_{SS}$ and $\phi_{S2S}$, but might have too heavy tails for some partially nonergodic parameters.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
## }
## 
## transformed data {
##   real h = 6;
##   real vref = 400;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> phi;
##   real<lower=0> tau;
##   
##   vector[NEQ] deltaB;
## }
## 
## model {
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   
##   phi ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   
##   for(i in 1:N) {
##     real mu;
##     mu = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]];
##     Y[i] ~ normal(mu,phi);
##   }
## }
```

A more efficient version of the above Stan code is provided below.
This version is vectorized and uses matrix algebra, which makes it run faster.
Note that this efficiency is not related to the use of vectorization _per se_, as the Stan program is compiled to **C++** before being called from Stan.
The increase in efficiency is more related to how the derivatives of the likelihood function are generated.
The NUTS sampler used by Stan requires derivatives of the likelihood function to be computed and efficiency gains in Stan are usually related to how to optimally represent the derivative tree required as part of this process.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
## }
## 
## transformed data {
##   real h = 6;
##   real vref = 400;
##   
##   matrix[N,7] X;
##   
##   for(i in 1:N) {
##     X[i,1] = 1;
##     X[i,2] = M[i];
##     X[i,3] = square(8 - M[i]);
##     X[i,4] = log(R[i] + h);
##     X[i,5] = M[i] * log(R[i] + h);
##     X[i,6] = R[i];
##     X[i,7] = log(VS[i]/vref);
##   }
## }
## 
## parameters {
##   vector[7] theta;
##   
##   real<lower=0> phi;
##   real<lower=0> tau;
##   
##   vector[NEQ] deltaB;
## }
## 
## model {
##   theta ~ normal(0,10);
##   
##   phi ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   
##   Y ~ normal(X * theta + deltaB[idx_eq],phi);
## }
```

Below, we show an example of how to define user-defined functions.
Here, we have defined the median prediction as a function that takes as input the vector of coefficients, the predictor variables, and he fixed parameters.
The function is declared as `real`, meaning that the return variable is of type real.
One could just as easily declare a function that takes as input the vector of coefficients, the matrix of predictor variables, and returns a vector of median predictions.


```
## functions {
##   real f_median(vector theta, real M, real R, real VS, real h, real vref) {
##     real mu = theta[1] + theta[2] * M + theta[3] * square(8 - M) + (theta[4] + theta[5] * M) * log(R + h) + theta[6] * R + theta[7] * log(VS/vref);
##     return mu;
##   }
## }
## 
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
## }
## 
## transformed data {
##   real h = 6;
##   real vref = 400;
## }
## 
## parameters {
##   vector[7] theta;
##   
##   real<lower=0> phi;
##   real<lower=0> tau;
##   
##   vector[NEQ] deltaB;
## }
## 
## model {
##   theta ~ normal(0,10);
##   
##   phi ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   
##   for(i in 1:N) {
##     real mu;
##     mu = f_median(theta, M[i], R[i], VS[i], h, vref) + deltaB[idx_eq[i]];
##     Y[i] ~ normal(mu,phi);
##   }
## }
```

# More Complicated Models

An obvious extension to the simple model is to include a random systematic effect for stations, and to make the near-fault-saturation term, $h$, a parameter of the model.
Estimating $h$ makes the model nonlinear, so we cannot use the fully vectorized model anymore.
The Stan code is below.
We added the number of stations and a station index to the data, and moved `h` from the transformed data block to the parameters block.
Since `h` should be positive, we declare it with `<lower=0>`.
The median is assigned in a loop because the functional form is nonlinear, but the sampling statement `Y ~ normal(mu,sigma)` is vectorized, which requires that `mu` is declared as a vector.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   real vref = 400;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
## }
## 
## model {
##   vector[N] mu;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```

Often, `h` is modeled as being magnitude dependent, such as $h = a \exp(bM)$.
This can be easily incorporated.
Since `a` and `b` are typically hard to estimate from data, they should be given informative priors.

```{}

parameters {
...
  real a;
  real b;
}

model {}
...
  for(i in 1:N) {
    mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + a * exp(b * M[i])) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
  }
  Y ~ normal(mu,phiSS);
}
```

So far, we defined all predictor variables (magnitude, distance, $V_{S30}$) as vectors of size `N`.
Since each earthquake is asscociated with one unique magnitude value, and each station with a unique $V_{S30}$ value, oen can also declare those variables as vectors (or arrays) of length `NEQ` or `NSTAT`, and then use the corresponding index to address their values.
This is useful when dealing with missing or uncertain values of the predictor variables.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[NEQ] M; // magnitudes
##   vector[N] R; // distances
##   vector[NSTAT] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   real vref = 400;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
## }
## 
## model {
##   vector[N] mu;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[idx_eq[i]] + theta3 * square(8 - M[idx_eq[i]]) + (theta4 + theta5 * M[idx_eq[i]]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[idx_stat[i]]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```

## Nonlinear Site Amplification

For soft soil sites, ground-motion depends on the input motion on base rock.
The following model shows how to model this effect in **Stan**.
Here, we have replaced the simple (linear) site amplification modle of the previous models wih the one of @Chiou2014.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[NEQ] M; // magnitudes
##   vector[N] R; // distances
##   vector[NSTAT] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   vector[NSTAT] lnVS;
##   vector[NSTAT] VS2;
## 
##   for(i in 1:NSTAT) {
##     lnVS[i] = fmin(log(VS[i] / 1130), 0.);
##     VS2[i] = fmin(VS[i], 1130.);
##   }
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
## 
##   real phi1;
##   real<upper=0> phi2;
##   real<upper=0> phi3;
##   real<lower=0> phi4;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
## }
## 
## model {
##   vector[N] mu;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   h ~ normal(6,4);
## 
##   phi2 ~ normal(0,1);
##   phi3 ~ normal(0,1);
##   phi4 ~ normal(0,1);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   for(i in 1:N) {
##     real yref = theta1 + theta2 * M[idx_eq[i]] + theta3 * square(8 - M[idx_eq[i]]) + (theta4 + theta5 * M[idx_eq[i]]) * log(R[i] + h) + theta6 * R[i] + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##     real fsite = phi1 * lnVS[idx_stat[i]];
##     real fnl = phi2 * (exp(phi3 * (VS2[idx_stat[i]] - 360)) - exp(phi3 * 770)) * log((exp(yref) + phi4) / phi4);
##     mu[i] = yref + fsite + fnl;
##   }
##   Y ~ normal(mu,phiSS);
## }
```

## Heteroscedastic models

In some GMMs, the between-event standard deviation $\tau$ is modeled as magnitude-dependent, with lower values for larger magnitudes.
The can be implemented by declaring `tau` as a vector of length `NEQ` (i.e. one value for each event), whose entries follow a function of magnitude.
An example is shown below.
It would be possible to declare the break points as a parameter, but this would cause problems in the automatic differentiation of the log density.
If one wants to estimate the transition from larger to smaller values of `tau`, it is better to declare the functional dependence for example as a logistic function.


``` {}
...
parameters {
  ...
  real<lower=0> tau1;
  real<lower=0> tau2;

}

transformed parameters {
  vector[NEQ] tau;
  
  for(i in 1:NEQ) {
    if(M[i] < 5)
      tau[i] = tau1;
    else if (M[i] < 6)
      tau[i] = tau1 + (tau2 - tau1) * (M[i] - 5);
    else
      tau[i] = tau2;
  }
}

model {
  ...
  deltaB ~ normal(0,tau);
}
```

## Correlated Random Effects

Below, we show an example of how one could model multiple random effects for the same level.
We assume that for each event, there is an event-specific near-fault-saturation term, which is distributed around a mean function $a \exp (b M)$, i.e.
$$
h_e = a \exp(b M_e + \delta B_{2,e})
$$
First, we just add a new variable for this term to the model.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   real vref = 400;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real a;
##   real b;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   real<lower=0> tau_h;
##   
## 
##   vector[NEQ] deltaB;
##   vector[NEQ] deltaB2;
##   
##   vector[NSTAT] deltaS;
## }
## 
## model {
##   vector[N] mu;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaS ~ normal(0,phiS2S);
##   deltaB ~ normal(0,tau);
##   deltaB2 ~ normal(0,tau_h);
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + a * exp(b * M[i] + deltaB2[idx_eq[i]])) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```

Next, we model the two event terms as correlated, _i.e._, distributed according to a multivariate normal distribution
$$
\vec{\delta B} \sim N(\vec{0},\boldsymbol{\Sigma})
$$
Thus, we now declare the event terms `deltaB` as an array of length `NEQ` of two-dimensional vectors, corresponding to the constant random effect and the event-specific near-fault-term.
The prior for the covariance matrix $\Sigma$ is separated into a prior for the standard deviations `tau` and one for the correlation marix `C_eq`, which is based on [@Lewandowski2009].
These are combined into the covariance matrix via `Sigma_eq = quad_form_diag(C_eq,tau)`, where `quad_form_diag(C_eq,tau) = diag_matrix(tau) * C_eq * diag_matrix(tau)`.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   real vref = 400;
##   vector[2] mu_eq;
##   
##   for(i in 1:2)
##     mu_eq[i] = 0;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real a;
##   real b;
##   
##   real<lower=0> phiSS;
##   vector<lower=0>[2] tau;
##   real<lower=0> phiS2S;
##   
##   corr_matrix[2] C_eq;
##   vector[2] deltaB[NEQ];
##   
##   vector[NSTAT] deltaS;
## }
## 
## model {
##   vector[N] mu;
##   matrix[2,2] Sigma_eq;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaS ~ normal(0,phiS2S);
##   
##   C_eq ~ lkj_corr(2);
##   Sigma_eq = quad_form_diag(C_eq,tau);
##   deltaB ~ multi_normal(mu_eq,Sigma_eq);
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + a * exp(b * M[i] + deltaB[idx_eq[i],2])) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i],1] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```

This model can also be coded in a different way, based on the Cholesky factorization of the correlated event terms, which should be more efficient.
In his case, we define a Cholesky factor `L_eq` as a parameter, which is given an LKJ prior as for the correlation matrix in the previous model.
The random effects for each are calculated by multiplying the Cholesky factor of the covariance matrix with a two-dimensional vector whose entries are distributed according to a standard normal distribution.
The Cholesky factor of the covariance is calculated by `diag_pre_multiply(tau, L_eq)`.



```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   real vref = 400;
##   vector[2] mu_eq;
##   
##   for(i in 1:2)
##     mu_eq[i] = 0;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real a;
##   real b;
##   
##   real<lower=0> phiSS;
##   vector<lower=0>[2] tau;
##   real<lower=0> phiS2S;
##   
##   cholesky_factor_corr[2] L_eq;
##   matrix[2,NEQ] z_eq;
##   
##   vector[NSTAT] deltaS;
## }
## 
## transformed parameters {
##   matrix[2,NEQ] deltaB;
## 
##   deltaB = diag_pre_multiply(tau, L_eq) * z_eq;
## 
## }
## 
## model {
##   vector[N] mu;
##   matrix[2,2] Sigma_eq;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaS ~ normal(0,phiS2S);
##   
##   L_eq ~ lkj_corr_cholesky(2);
##   to_vector(z_eq) ~ std_normal();
##   
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + a * exp(b * M[i] + deltaB[2,idx_eq[i]])) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[1,idx_eq[i]] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```

## Partially Nonergodic Models

Over the last few years it has been recognized that the ergodic assumption [@Anderson1999] (that the ground-motion distribution at a site over time is the same as the ground-motion distribution over space) can lead to biased hazard results.
With an increasing amount of data in different regions, the ergodic assumption can be relaxed.
An intermediate step towards fully nonergodic models are partially nonergodic models (though one can argue that models that account for systematic station terms $\delta S$ are already partially nonergodic) in which some of the parameters are different for different regions.
Often, the constant, anelastic attenuation coefficient, and the site-scaling coefficient are regionally dependent [@Stafford2014; @Kotha2016; @Kuehn2016; @Sedaghati2017].
It makes sense to model these as regional random effects, since in that case the coefficient for regions with a smaller amount of data are automatically associated with larger uncertainty (in the Bayesian case, a wider posterior distribution).
Typically, one assumes that the regional random effects are distributed according to a normal distribution, where the regional coefficients are samples from a global coefficient (_e.g._, for the constant $\theta_1$)
$$
\theta_1 \sim \mathcal N(\mu_{\theta 1},\sigma_{\theta 1})
$$
where $\mu_{\theta_1}$ is the (global) mean (over the data set) for the constant coefficient and $\sigma_{\theta_1}$ is the standard deviation which determines how much the regional coefficients can differ from the global mean.
Below, we have written a Stan model with regional coefficients for the constant, the anelastic attenuation (linear R scaling) and the $V_{S30}$-scaling.
We add an integer for the number of regions, as well as an integer comprising the region indices for the records to the `data {}` block.
We then declare the mean coefficients and vectors for the regional coefficients, as well as the standard deviations.
In this case, the regional parameters are declared as independent, but they can also be modeled as correlated as explained previously.
We have declared the regional parameters `theta6` (linear R term) with `vector<upper=0>[NREG] theta6`, so they are constrained to be negative.
This is a physical requirement, but in particular for partially nonergodic models it can happen that a regional coefficient becomes positive if data is sparse - this can also happen for long periods, where the coefficient typically approaches zero.
Imposing the constraint ensures that this does not happen.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   int<lower=1> NREG; // number of regions
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
##   int<lower=1,upper=NREG> idx_reg[N];
## }
## 
## transformed data {
##   real vref = 400;
## }
## 
## parameters {
##   real mu_theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real mu_theta6;
##   real mu_theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   real<lower=0> sigma_theta1;
##   real<lower=0> sigma_theta6;
##   real<lower=0> sigma_theta7;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
##   
##   vector[NREG] theta1;
##   vector<upper=0>[NREG] theta6;
##   vector[NREG] theta7;
## }
## 
## model {
##   vector[N] mu;
## 
##   mu_theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   mu_theta6 ~ normal(0,10);
##   mu_theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   sigma_theta1 ~ cauchy(0,0.5);
##   sigma_theta6 ~ cauchy(0,0.01);
##   sigma_theta7 ~ cauchy(0,0.3);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   theta1 ~ normal(mu_theta1,sigma_theta1);
##   theta6 ~ normal(mu_theta6,sigma_theta6);
##   theta7 ~ normal(mu_theta7,sigma_theta7);
##   
##   for(i in 1:N) {
##     mu[i] = theta1[idx_reg[i]] + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6[idx_reg[i]] * R[i] + theta7[idx_reg[i]] * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```

The model can be rewritten, using a so-called non-centered parameterization, by recognizing that
$$
\theta_1 \sim \mathcal N(\mu_{\theta 1},\sigma_{\theta 1})
$$
is the same as
$$
\theta_1 = \mu_{\theta 1} + z \sigma_{\theta 1}\\
z \sim \mathcal N(0,1)
$$
Hence, we now declare a vector `z` of length `NREG` for each regionally varying coefficient, which has a standard normal prior distribution, and calculate the parameters in the `transformed parameters {}` block according to the above equation.
To ensure that `theta6` is positive, we updated the upper limit for `z6` (this also serves as an example that parameters can be used as upper/lower limits).
The rest of the model is the same.
The non-centered parameterization changes the geometry of the posterior distribution, and can help to avoid problems that arise due to correlations between paramters [@Betancourt2013].

One can also model the regional parameters as correlated, similar to he model with correlated event terms above.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   int<lower=1> NREG; // number of regions
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
##   int<lower=1,upper=NREG> idx_reg[N];
## }
## 
## transformed data {
##   real vref = 400;
## }
## 
## parameters {
##   real mu_theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real mu_theta6;
##   real mu_theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   real<lower=0> sigma_theta1;
##   real<lower=0> sigma_theta6;
##   real<lower=0> sigma_theta7;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
##   
##   vector[NREG] z1;
##   vector<upper=-mu_theta6/sigma_theta6>[NREG] z6;
##   vector[NREG] z7;
## }
## 
## transformed parameters {
##   vector[NREG] theta1;
##   vector[NREG] theta6;
##   vector[NREG] theta7;
##   
##   theta1 = mu_theta1 + z1 * sigma_theta1;
##   theta6 = mu_theta6 + z6 * sigma_theta6;
##   theta7 = mu_theta7 + z7 * sigma_theta7;
## }
## 
## model {
##   vector[N] mu;
## 
##   mu_theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   mu_theta6 ~ normal(0,10);
##   mu_theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   sigma_theta1 ~ cauchy(0,0.5);
##   sigma_theta6 ~ cauchy(0,0.01);
##   sigma_theta7 ~ cauchy(0,0.3);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   z1 ~ std_normal();
##   z6 ~ std_normal();
##   z7 ~ std_normal();
##   
##   for(i in 1:N) {
##     mu[i] = theta1[idx_reg[i]] + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6[idx_reg[i]] * R[i] + theta7[idx_reg[i]] * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```

I newer versions of **Stan**, the non-centered parameterization can be coded simpler, by declaring the offset $\mu_\theta$ and multiplier $\sigma_\theta$ in the declaration of the regional parameters.
This avoids the need to declare a standard normal variable $z$, and explicitly calculating the regional parameters.
An example is shown in the next model.
As of **Stan** version 2.24.1, it is not possible to use the multiplier/offset declaration together with an upper or lower bound.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   int<lower=1> NREG; // number of regions
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
##   int<lower=1,upper=NREG> idx_reg[N];
## }
## 
## transformed data {
##   real vref = 400;
## }
## 
## parameters {
##   real mu_theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real mu_theta6;
##   real mu_theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   real<lower=0> sigma_theta1;
##   real<lower=0> sigma_theta6;
##   real<lower=0> sigma_theta7;
## 
##   vector<offset = mu_theta1, multiplier = sigma_theta1>[NREG] theta1;
##   vector<offset = mu_theta6, multiplier = sigma_theta6>[NREG] theta6;
##   vector<offset = mu_theta7, multiplier = sigma_theta7>[NREG] theta7;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
##   
## }
## 
## model {
##   vector[N] mu;
## 
##   mu_theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   mu_theta6 ~ normal(0,10);
##   mu_theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   sigma_theta1 ~ cauchy(0,0.5);
##   sigma_theta6 ~ cauchy(0,0.01);
##   sigma_theta7 ~ cauchy(0,0.3);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   for(i in 1:N) {
##     mu[i] = theta1[idx_reg[i]] + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6[idx_reg[i]] * R[i] + theta7[idx_reg[i]] * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```


In the previous partially nonergodic models, the prior distribution for the regional standard deviations `sigma_theta1`, `sigma_theta6` and `sigma_theta7` was a half-cauchy distribution.
This is based on a recommendation by [@Gelman2006b].
However, the half-cauchy distribution has heavy tails, which can lead to unrealistically high values of the standard deviation and thus spurious regional deviations from the global mean coefficient (the deviation is $z \sigma_{\theta}$) if the number of regions is small.
This is also a problem of maximum-likelihood estimation with a small number of groups.
For event terms and station terms, this is less of a problem since there are many events/stations for each data set, but the number of regions is typically small (<10), which can make it hard to estimate the regional standard deviation.
In that case, stronger prior information is needed - for example, an exponential distribution or a normal or Student-t distribution might be better.
These can be implemented as

``` {}
  sigma_par1 ~ exponential(1);
  sigma_par1 ~ normal(0,1);
  sigma_par1 ~ student_t(6,0,1);
```

What is generally important is that the prior distribution should be scaled based on the effects.
This is a complicated topic, and general advice that works in every situation is difficult.


## Robust Regression

Sometimes a data set has outliers, and these outliers can severley affect both the mean and standard deviation ($\phi$) of an estimated model - often, GMPE developers discard some records which are obvious outliers (these might be of low quality due to processing errors).
One way to mitigate the effect of outlier data points on the model is to use robust regression - in robust regression, one tries to limit the influence of a data point that is far from the regression line.
For example, one way to do that would be to minimize the absolute residuals and not the squared residuals.
In a Bayesian model, one way to do robust regression is to replace the data likelihood (which until now has been based upon the assumption that logarithmic ground-motions are normally distributed, when conditioned upon some rupture scenario) with a Student-_t_ distribution with low degrees-of-freedom.
Such a distribution has heavier tails than the normal distribution and thus is less sensitive to outliers.
Such a model can be coded in Stan as follows (see also <http://doingbayesiandataanalysis.blogspot.com/2013/06/bayesian-robust-regression-for-anscombe.html> or <https://jrnold.github.io/bayesian_notes/robust-regression.html>)

``` {}
...
parameters {
  real<lower=1> nu;
}
model {
  ...
  nu ~ gamma(2,0.1);
  ...

  Y ~ student_t(nu,mu,sigma);
}
...
```

Here, we declare a new parameter `nu` for the degrees-of-freedom, which is given a gamma prior (following [@Juarez2010]).
The rest of the model is the same, except that the sampling statement for the data is changed from the normal distribution to the Student-_t_ distribution.


## Multiple Target Variables

The models so far deal with one target variable.
Each of these can be run multiple times for different target variables; this is common practicec in GMM development, where regressions are run independently for different periods of PSA.
In general, PSA at different periods are correlated (e.g. [@Bradley2012,@Baker2008b]).
These correlations can be directly modeled in the regression [@Kuehn2015].
The following model describes how that can be done.

Here, we define the number of periods (target variables) we want to model as `NP`.
The target variable is now declared as an array of vectors of length `NP`, i.e. `vector[NP] Y[N];`.
Correspondingly, each parameter is declared as a vector of length `NP`, and event terms,station terms are arrays of vectors.
We also declare cholesky factors corresponding to the correlation of event terms, station terms, and within-event/within-station residuals.
We need to add a loop over the number of periods to calculate the median predictions for all periods.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   int<lower=1> NP; //  number of periods
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[NP] Y[N];       // log psa
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   real vref = 400;
##   vector[NP] mu = rep_vector(0,NP);
## }
## 
## parameters {
##   vector[NP] theta1;
##   vector[NP] theta2;
##   vector[NP] theta3;
##   vector[NP] theta4;
##   vector[NP] theta5;
##   vector[NP] theta6;
##   vector[NP] theta7;
##   
##   vector<lower=0>[NP] h;
##   
##   vector<lower=0>[NP] phiSS;
##   vector<lower=0>[NP] tau;
##   vector<lower=0>[NP] phiS2S;
## 
##   vector[NP] deltaB[NEQ];
##   vector[NP] deltaS[NSTAT];
## 
##   cholesky_factor_corr[NP] L_p;
##   cholesky_factor_corr[NP] L_eq;
##   cholesky_factor_corr[NP] L_stat;
## }
## 
## model {
##   vector[NP] mu_rec[N];
##   matrix[NP,NP] L_Sigma;
##   matrix[NP,NP] L_Sigma_eq;
##   matrix[NP,NP] L_Sigma_stat;
## 
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   L_p ~ lkj_corr_cholesky(1);
##   L_Sigma = diag_pre_multiply(phiSS, L_p);
## 
##   L_eq ~ lkj_corr_cholesky(1);
##   L_Sigma_eq = diag_pre_multiply(tau, L_eq);
## 
##   L_stat ~ lkj_corr_cholesky(1);
##   L_Sigma_stat = diag_pre_multiply(phiS2S, L_stat);
## 
##   deltaB ~ multi_normal_cholesky(mu,L_Sigma_eq);
##   deltaS ~ multi_normal_cholesky(mu,L_Sigma_stat);
##  
##   for(p in 1:NP) {
##     for(i in 1:N) {
##       mu_rec[i,p] = theta1[p] + theta2[p] * M[i] + theta3[p] * square(8 - M[i]) + (theta4[p] + theta5[p] * M[i]) * log(R[i] + h[p]) + theta6[p] * R[i] + theta7[p] * log(VS[i]/vref) + deltaB[idx_eq[i],p] + deltaS[idx_stat[i],p];
##     }
##   }
##   Y ~ multi_normal_cholesky(mu_rec,L_Sigma);
## }
```

## Missing Data

Sometimes, there is missing data for some records.
Typically in GMPE estimation, these records are ignored.
However, there are methods how one can deal with missing data [@Rubin1976a,@Allison2002].
A standard method is multiple imputation, for example implemented in the R-package `mi` [@Su2011] or `mice` [@VanBuuren2011].
In Stan (or Bayesian inference in general), one can declare a missing data point as a parameter in the model, which is estimated.
Since a posterior distribution is estimated for the missing data (with different samples), this functions as some sort of multiple imputation.

Below, we show an example, where we assume that some $V_{S30}$-values are missing.
Since each station has a unique $V_{S30}$-value, that means that we declare the input $V_{S30}$-values for each stations via `vector[N] VS;`.
Similar to the station term,this means that we also need the index connecting stations to records.
We also need a new index which comprises the indices of the missing values, as well as the number of stations with missing values.
For example, if the second and tenth station had an unknown $V_{S30}$-value, then the array of missing values would be of length 2 and consists of `c(2,10)`.

The missing values are declared as `vector[N_missing_VS] VS_missing;`.
Since we cannot assign new values to loaded data, we declare a new variable in the `transformed parameters {}` block, which contains the original `VS` values, and the estimated missing values for the correct indices.
This new variable is then used in the calcuation of the median prediction.

The missing values should be assigned an informative prior distribution.
If this is not available, one could use the (geometric) mean and standard deviation of the available $V_{S30}$-values.
Instead of estimating $V_{S30}$, one could also estimate $\ln V_{S30}$.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   int<lower=1,upper=NSTAT> N_missing_VS;
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[NSTAT] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
##   int<lower=1,upper=NSTAT> idx_missing_vs[N_missing_VS];
## }
## 
## transformed data {
##   real vref = 400;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
##   
##   vector<lower=0>[N_missing_VS] VS_missing;
## }
## 
## transformed parameters {
##   vector[NSTAT] VS_imputed = VS;
##   
##   for(i in 1:N_missing_VS)
##     VS_imputed[idx_missing_vs[i]] = VS_missing[i];
## }
## 
## model {
##   vector[N] mu;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   VS_missing ~ lognormal(log(400),0.5); // some prior should be set (if no information, maybe mean/sd of dataset)
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS_imputed[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```

## Measurement Error Models

In general, the predictor variables in a GMM are no known exactly, but associated with some error (or epistemic uncertainty).
This is typically not modeled, and leads o an overestimation of aleatory variability.
Measurement error models can be incorporated in **Stan** similar to the missing data model described previously.
For example, to account for uncertain $V_{S30}$-values, one can declare`VS` as a parameter to be estiamted for each station, with a prior distribution that is informed by the observed $V_{S30}$-value and its measurement uncertainty.
See @Kuehn2018 for a detailed description.

Below, we show an implementation of such a model in Stan for uncertain $V_{S30}$-values.
The input requires an observed $V_{S30}$-value (`vector[NSTAT] VS_obs;`) and a value of the logarithmic standard deviation (`vector[NSTAT] VS_sd;`).
The model assumes that
$$
\ln V_{S30,obs} \sim N(\ln V_{S30,*}, \sigma(\ln V_{S30}))
$$
where $V_{S30,*}$ is the ``true'' $V_{S30}$-value that is used in the calculation of the median.
It is important to note that results can be sensitive to $\sigma(\ln V_{S30})$, as larger values of this parameter allow the $V_{S30}$-values to vary more, which leads to a stronger reduction in $\phi_{S2S}$.
Similarly, large values of magnitude uncertainty can lead to a strong reduction in $\tau$, so these values should be assessed with care.

In the Stan model shown below, there is one target variable.
This means that, if repeated for different target variables (e.g. different periods of the response spectrum), one gets a different estimate of $V_{S30,*}$ for ach target variable.
To avoid this behavior, @Kuehn2018 combined the measurement error model with a model for multiple, correlated target variables.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of stations
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[NSTAT] VS_obs; // Vs30 values
##   vector[NSTAT] VS_sd; // standard deviation of log Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   real h = 6;
##   real vref = 400;
## 
##   vector[NSTAT] lnVS_obs = log(VS_obs);
##   real mean_lnVS = mean(lnVS_obs);
##   real<lower=0> sd_lnVS = sd(lnVS_obs);
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
##   
##   vector<lower=0>[NSTAT] lnVS_star;
## }
## 
## model {
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
## 
##   lnVS_star ~ normal(mean_lnVS,sd_lnVS);
##   lnVS_obs ~ normal(lnVS_star,VS_sd);
##   
##   for(i in 1:N) {
##     real mu;
##     mu = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * (lnVS_star[idx_stat[i]] - log(vref)) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##     Y[i] ~ normal(mu, phiSS);
##   }
## }
```

## Truncated Regression

If data comes from _e.g._, triggered instruments, it is truncated.
If this is not taken into account, the regression estimates will be biased (@Chao2018,@Bragato2004).
Truncated regression can be performed in Stan using the following adjustment to the sampling statement.
In this case, after he samplig statement we add `T(a,b)`, where a is the lower truncation bound, and `b` is the upper truncation bound.
For ground-motion data, one typically deals with lower truncation, so the upper truncation level is left empty, corresponding to no truncation.
A truncated sampling statement of the form `T(,)` cannot be vectorized, so it is moved into the loop (which unfortunately is less efficient).
However, lower truncation is just a simple adjustment to the likelihood of the form
$$\mathcal L_{trunc} = \frac{pdf(y | \mu, \sigma)}{1 - cdf(a | \mu, \sigma)}$$
This statement can be vecorized by using the `target += ...` notation.`
The difference is that when using the `T(,)` formulation, **Stan** checks at runtime that the values of `Y` do not fall outside the bounds, while no such check is performed using the `target += ` notation.
The vectorized formulation should be faster.


``` {}
data {
  ...
  real trigger_level;
}
...

model {
  ...
  // formulation using T(lower bound, )
  for(i in 1:N) {
    real mu;
    mu = ...;
    Y[i] ~ normal(mu, phiSS) T(trigger_level,);
  }
  
  // alternative formulation using target += 
  for(i in 1:N) {
    real mu;
    mu = ...;
  }
  target += normal_lpdf(Y | mu, phiSS) - normal_lccdf(trigger_levelc | mu, phiSS);
}
```

In general, ground-motion data is truncated on PGA.
This means that other ground-motion parameters are not sharply truncated, depending on their correlation with PGA.
@Kuehn2020a modeled the truncation of PSA at different by adjusting the joint likelihood based on the PGA trigger level.
This means that the target variables are model as correlated, and the adjustment is done for PGA (for details, see @Kuehn2020a).
An example model is shown below.
It should be noted that such a model can be slow to run, since calculating the cholesky decompositions is computationally expensive.


```
## data {
##   int<lower=1> N;  // overall number of records
##   int<lower=1> NEQ;   // number of earthquakes
##   int<lower=1> NSTAT; // number of stations
##   int<lower=1> NP;
## 
##   vector[N] M;       // distance
##   vector[N] R;       // distance
##   vector[NP] Y[N];       // log psa
## 
##   real triggerlevel;
## 
##   int<lower=1,upper=NEQ> idx_eq[N];       // earthquake id
##   int<lower=1,upper=NSTAT> idx_stat[N];     // station id
## }
## 
## transformed data {
##   vector[NP] mu = rep_vector(0,NP);
##   vector[N] M2 = square(8 - M);
##   vector[N] lnR = log(R + 6);
##   vector[N] MlnR = M .* log(R + 6);
## }
## 
## 
## parameters {
##   vector<lower=0>[NP] phiSS;
##   vector<lower=0>[NP] tau;
##   vector<lower=0>[NP] phiS2S;
## 
##   vector[NP] theta1;
##   vector[NP] theta2;
##   vector[NP] theta3;
##   vector[NP] theta4;
##   vector[NP] theta5;
##   vector<upper=0>[NP] theta6;
## 
##   vector[NP] deltaB[NEQ];
##   vector[NP] deltaS[NSTAT];
## 
##   cholesky_factor_corr[NP] L_p;
##   cholesky_factor_corr[NP] L_eq;
##   cholesky_factor_corr[NP] L_stat;
## }
## 
## model {
##   vector[NP] mu_rec[N];
##   matrix[NP,NP] L_Sigma;
##   matrix[NP,NP] L_Sigma_eq;
##   matrix[NP,NP] L_Sigma_stat;
## 
##   phiSS ~ normal(0,1);
##   tau ~ normal(0,1);
##   phiS2S ~ normal(0,1);
## 
##   theta1 ~ normal(0,5);
##   theta2 ~ normal(0,5);
##   theta3 ~ normal(0,5);
##   theta4 ~ normal(0,5);
##   theta5 ~ normal(0,5);
##   theta6 ~ normal(0,0.01);
## 
##   L_p ~ lkj_corr_cholesky(1);
##   L_Sigma = diag_pre_multiply(phiSS, L_p);
## 
##   L_eq ~ lkj_corr_cholesky(1);
##   L_Sigma_eq = diag_pre_multiply(tau, L_eq);
## 
##   L_stat ~ lkj_corr_cholesky(1);
##   L_Sigma_stat = diag_pre_multiply(phiS2S, L_stat);
## 
##   deltaB ~ multi_normal_cholesky(mu,L_Sigma_eq);
##   deltaS ~ multi_normal_cholesky(mu,L_Sigma_stat);
## 
##   for(p in 1:NP) {
##     for(i in 1:N) {
##       mu_rec[i,p] = theta1[p] + theta2[p] * M[i] +  theta3[p] * M2[i] + theta4[p] * lnR[i] + theta5[p] * MlnR[i] + theta6[p] * R[i] + deltaB[idx_eq[i],p] + deltaS[idx_stat[i],p];
##     }
##   }
## 
##   target += multi_normal_cholesky_lpdf(Y | mu_rec, L_Sigma) - normal_lccdf(triggerlevel | mu_rec[:,1], phiSS[1]);
## }
```

## Different Data Likelihoods (Mixture Models)

In most of the previous examples, the data was assumed to be distributed lognormally (or more precisely, the target variable was the logarithm of the ground-motion parameter of interest and was modeled as normally distributed).
This is a general (and typically reasonable) assumption in the estimation of GMPEs.
However, it is not necessary to use a normal data likelihood (coded as `Y ~ normal(mu,sigma);`).
We have already seen how the normal distribution can be replaced with a Student-_t_ distribution.
One can also use other distributions.
For example, in some hazard studies (SWUS, Hanford) the ground-motion distribution is modeled as a mixture of two normal distributions (more accurately, a branch in the logic tree for the standard deviation contains the mixture distribution).
Typically, the components of the mixture distribution (weights and standard deviations) are estimated from residuals, but it raises the question whether the coefficients of a GMPE would be different if a mixture model is used in the estimation.
This can be done in Stan, though the implementation is a bit more complicated than what has been used previously.

Similarly, the mixture model can be used to combine different GMPEs into a single model as in @Haendel2014.

A GMPE mixture model as used in SWUS and Hanford has the form
$$
Y = f(M,R,\ldots) + \delta B + \delta S + \delta WS \\
\delta B \sim \mathcal N(0,\tau) \\
\delta S \sim \mathcal N(0,\phi_{S2S}) \\
\delta WS \sim \mathcal w_1 N(0,\phi_{SS,1}) + w_2 \mathcal N(0,\phi_{SS,2}) \\
w_1 + w_2 = 1
$$

where $w_1$ and $w_2$ are the weights for the mixture components, and $\phi_{SS,1}$ and $\phi_{SS,2}$ are the standard deviations (we follow the SWUS and Hanford project and assume that the mean of the two normal distributions is zero, but in general that does not have to be the case).
One could easily generalize this formulation to include more components for the mixture distribution.

Mixture models are in general difficult to estimate via MCMC, since they are not identifiable (one could swap the indices of $w$ and $\phi_{SS}$ and end up with the same likelihood), which means that different chains can have swapped estimated parameters, even though they have converged.
Sometimes, yhis can happen within a chain.

A generative model to sample from a mixture model would be to sample an index (in this case 1 or 2) with weights `wt`, and then sample from the component of the mixture with the sampled index.
This approach cannot be used in the implementation as a forward model in Stan, because Stan cannot sample from integer distributions.
Instead, the mixture proportions need to be integrated out

To implement the mixture model in Stan, we need to declare the two standard deviations as well as the weights.
The weights are declared as `simplex[2] wt;` -- this declares `wt` as a vector of length 2 with the constraint that the components sum up to one.
In the model, we do not specify a prior distribuion for the weights, in which case Stan automatically assings a uniform prior over the declared range (in this case from zero to one).

In the likelihood, the log-probability is incremented by the sum of the log probabilities of the two components.
The density of the mixture has form
$$
p(y|w_1,w_2,\mu,\phi_{SS,1},\phi_{SS,2}) = w_1 \mbox{normal}(y|\mu,\phi_{SS,1}) + w_2 \mbox{normal}(y|\mu,\phi_{SS,2})
$$
where $\mu = f(M,R,\ldots) + \delta B + \delta S$ (in the formulation above, both mixture components have the same median).
The log-probability in the mixture model in Stan is incremented by the logarithm of this density.
It makes use of the function `log_sum_exp`, which is defined as `log_sum_exp(a,b) = log(exp(a) + exp(b))`.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## }
## 
## transformed data {
##   real vref = 760;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS_1;
##   real<lower=0> phiSS_2;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
##   
##   simplex[2] wt;
## }
## 
## model {
##   vector[N] mu;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS_1 ~ cauchy(0,0.5);
##   phiSS_2 ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##     
##     target += log_sum_exp(log(wt[1]) + normal_lpdf(Y[i] | mu[i], phiSS_1), log(wt[2]) + normal_lpdf(Y[i] | mu[i], phiSS_2));
##   }
## }
```

# Spatial Models

## Within-Event Correlation

Ground-motion observations made at closely-spaced locations will be correlated by virtue of the fact that they sample similar travel paths from source to site.
This means that directionality effects from the source will be similar and that near-surface crustal amplification effects will also be similar.
The models considered thus far do not account for this.
Studies like @Jayaram2009a derive spatial correlation models of the form:
$$
  \rho(\boldsymbol{x}_i,\boldsymbol{x}_j) = \exp\left( -\frac{||\boldsymbol{x}_i-\boldsymbol{x}_j||}{r_c}\right)
$$
using within-event residuals computed from ground-motion models.
Within-event residuals $\boldsymbol{\varepsilon}(\boldsymbol{x})$ are computed for all observations made for each event and the spatial correlation among these observations is well represented by the exponential model shown above that depends upon a single parameter $r_c$ which is the correlation length.
@Jayaram2010a showed how to account for some known degree of spatial correlation within the traditional random-effects regression framework of [@Abrahamson1992] when only a random effect for each event is considered.
The extension to more elaborate cases where multiple random effects are considered is more challenging to deal with, but can be handled using `stan` [@Stafford2018a].

Note that @Jayaram2009a work with within-event residuals, but not site-corrected within-event residuals. 
As a result, they find that the correlation length $r_c$ differs from region-to-region depending upon the degree of correlation among sites with similar levels of $V_{S,30}$.
In a model where random effects are considered for each site we should expect less spatial correlation as the site random effects will be spatially correlated and this portion of the correlation is no longer contained in the site-corrected within-event residuals.
In `stan`, random effects for site, or for event, can also be regarded as being spatially-correlated, but in the example below we focus upon the representation of the site-corrected within-event residuals.

In the previously considered models the covariance matrix of the within-event residuals was equivalent to:
$$
  \boldsymbol{\Sigma_\varepsilon} = \phi_{SS} \boldsymbol{I}_{n}
$$
where $\boldsymbol{I}_n$ is an $n\times n$ identity matrix.
When spatial correlations are considered this changes to:
$$
  \boldsymbol{\Sigma_\varepsilon} = \phi_{SS} \boldsymbol{\Lambda}_{n}(\boldsymbol{x};r_c)
$$
where $\boldsymbol{\Lambda}_{n}(\boldsymbol{x};r_c)$ is an $n\times n$ correlation matrix that is obtained by computing the exponential correlation as a function of inter-station distances and the correlation length $r_c$.
However, as we only consider the spatial correlations within each event we have a block-diagonal structure for $\boldsymbol{\Lambda}_{n}(\boldsymbol{x};r_c)$.
$$
  \boldsymbol{\Lambda}_{n}(\boldsymbol{x};r_c) \equiv \boldsymbol{\Lambda}_{n_{rec}^{(1)}}(\boldsymbol{x};r_c,eq^{(1)}) \oplus \boldsymbol{\Lambda}_{n_{rec}^{(2)}}(\boldsymbol{x};r_c,eq^{(2)}) \oplus \ldots \oplus \boldsymbol{\Lambda}_{n_{rec}^{(n_{eq})}}(\boldsymbol{x};r_c,eq^{(n_{eq})})
$$
where $\oplus$ is the direct sum operator, and the $\boldsymbol{\Lambda}_{n_{rec}^{(i)}}$ represents the within-event covariance matrix for earthquake $i$. 
This within-event covariance matrix is of size $n_{rec}^{(i)}\times n_{rec}^{(i)}$, with $\sum_{i}^{n_{eq}} n_{rec}^{(i)} = n$

Note that this block diagonal structure holds only for the event- and site-corrected within-event residuals.
The inclusion of random effects for events also has a block diagonal structure of the same form as the spatial covariance matrix $\boldsymbol{\Lambda}(\boldsymbol{x};r_c)$, but considering random effects for sites adds elements that _link_ blocks by virtue of the fact that the same sites record multiple events.

A current limitation of `stan` is that it does not have the ability to handle _ragged_ data structures.
That is, we cannot define an array that hold arrays or matrices of differing size as its elements.
When working with correlations among the observations of each event such a data structure would be very useful as it would allow us to define inter-station distance matrices for each event seperately.
Instead we can have a few options:  
* manually define $n_{eq}$ distance matrices for each event (either passed to `stan` as data, or created within `stan` from station coordinates);  
* work with a single very large $n\times n$ distance, and hence correlation and covariance matrix; or,  
* have some large $n\times n$ distance matrix, but only work with event-specific blocks for the purposes of sampling  

The first of these options, while ultimately the most computationally efficient, does not scale well to analyses that consider a large number of events. 
The second option, while the easiest to implement programatically, is the least computationally efficient.
Generally speaking, matrix operations (like Cholesky factorisation, matrix inversion, _etc_) tend to scale as $\mathcal{O}(n^3)$.
Therefore, the third option of accessing sub-blocks tends to be the preferred option currently.
Accessing sub-blocks of the matrix requires a local copy to be made which has a cost associated with it, but this is offset by enabling the use of smaller $n$ for each matrix operation.
The example code below shows the use of this latter approach.

### Known spatial correlation length

This code corresponds to a case in which the correlation length is assumed known ahead of the regression analysis and so the spatial correlation matrix can be passed with its block-diagonal structure as data.
Also passed is `ObsPerEvent` which is an array of length `NEQ` defining how many records each event has.
This array is used to define indexing into the vectors of observations and means as well as the covariance matrix using calls such as:
```{}
...
vector[ObsPerEvent[i]] eventMu = segment(mu, epos, ObsPerEvent[i]);
matrix[ObsPerEvent[i],ObsPerEvent[i]] eventLCov = phiSS * block( LspatialCorrelationMatrix, epos, epos, ObsPerEvent[i], ObsPerEvent[i] );
...
```
where `segment(x, i, n)` extracts a portion of an array `x` starting at index `i` and inclusively taking the `n` consecutive elements.
Similarly, `block(X, i, j, ni, nj)` extracts a sub-matrix from `X` starting from index `(i,j)` and taking `ni` elements in the first dimension and `nj` elements in the second dimension.
In order for this approach to work, the observations need to be sorted so that they are grouped by event and that this indexing operations will be accessing contiguous data corresponding to each event.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
##   
##   matrix[N,N] spatialCorrelationMatrix; // within-event correlation matrix (block diagonal matrix)
##   int<lower=1,upper=NEQ> ObsPerEvent[NEQ];
## }
## 
## transformed data {
##   real vref = 760;
##   // taking Cholesky here is only valid if spatial correlation matrix is block diagonal
##   cholesky_factor_corr[N] LspatialCorrelationMatrix = cholesky_decompose(spatialCorrelationMatrix);
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
## }
## 
## model {
##   vector[N] mu;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S);
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
## 
##   {
##     int epos;
##     epos = 1;
##     for ( i in 1:NEQ ) {
##       if ( ObsPerEvent[i] > 1 ) {
##         vector[ObsPerEvent[i]] eventMu = segment(mu, epos, ObsPerEvent[i]);
##         matrix[ObsPerEvent[i],ObsPerEvent[i]] eventLCov = phiSS * block( LspatialCorrelationMatrix, epos, epos, ObsPerEvent[i], ObsPerEvent[i] );
##         segment(Y, epos, ObsPerEvent[i]) ~ multi_normal_cholesky( eventMu, eventLCov );
##       } else {
##         real eventMu = mu[epos];
##         Y[epos] ~ normal( eventMu, phiSS );
##       }
##       epos = epos + ObsPerEvent[i];
##     }
##   }
## }
```

### Unknown spatial correlation length

In the case above we already knew the spatial correlation and so could compute the cholesky factor of the spatial correlation matrix within the `transformed data` block.
In the case where we wish to solve for the spatial correlation length, things are not so simple.
Now, as the correlation length will vary as a parameter of the model, the correlation matrix for each event will also be varying with every sample.
The only attribute that remains fixed is the inter-station distance matrix.
The code below demonstrates this case.
Here we pass in the distance matrix along with the `ObsPerEvent` array.
The event-specific mean and covariance matrices are computed _on the fly_ within the `model` block.
This approach is far less efficient than the case where the spatial correlation is known in advance.
Note that for the particular example considered here, it is not a good idea to try to derive the correlation length in this manner. 
The example model being considered is too simplistic to adequately represent the scaling for all source-to-site combinations and so systematic biases will exist that become mapped into apparent spatial correlations.
Generally speaking, this is also true for more detailed analyses as the ergodic mixing of the dataset leads to biases from event-to-event that appear as apparent correlations and extend estimates of the correlation length.
Note also that studies like [@Jayaram2009a] also preferentially focus upon short inter-station separation distances when calibrating their correlation model as these distances are of greatest importance in practice and the consideration of larger separation distances tends to bias the computed correlation lengths.
That said, region-specific studies where separation distances among stations are not large [@Stafford2018a] have obtained consistent results using both approaches.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
##   
##   matrix[N,N] distanceMatrix; // inter-station distance matrix (for all records across all events)
##   int<lower=1,upper=NEQ> ObsPerEvent[NEQ];
## }
## 
## transformed data {
##   real vref = 760;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S_0;
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
## 
##   real<lower=0> correlationLength;
## }
## 
## model {
##   vector[N] mu;
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S_0 ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S_0);
##   
##   correlationLength ~ cauchy(5,1);
## 
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
##   }
## 
##   {
##     int epos;
##     epos = 1;
##     for ( i in 1:NEQ ) {
##       if ( ObsPerEvent[i] > 1 ) {
##         vector[ObsPerEvent[i]] eventMu = segment(mu, epos, ObsPerEvent[i]);
##         matrix[ObsPerEvent[i],ObsPerEvent[i]] eventLCov = phiSS * cholesky_decompose( exp( -block(distanceMatrix, epos, epos, ObsPerEvent[i], ObsPerEvent[i] ) / correlationLength ) );
##         segment(Y, epos, ObsPerEvent[i]) ~ multi_normal_cholesky( eventMu, eventLCov );
##       } else {
##         real eventMu = mu[epos];
##         Y[epos] ~ normal( eventMu, phiSS );
##       }
##       epos = epos + ObsPerEvent[i];
##     }
##   }
## }
```

## Spatial Correlation of Station Terms

In the previous section, we described a model that takes spatial correlation of within-event/within-station residuals into account.
As noted, site terms (as well as event terms) should also be spatially correlated.
In this section, we describe how to take into account spatial correlations of site terms.

Here, we assume that site terms $\vec{\delta S}$ are distributed according to a multivariate normal distribution
$$
\vec{\delta S} \sim N(\vec{0}, \Phi_{S2S})
$$
The covariance matrix $\Phi_{S2S}$ is the sum of a matrix accounting for the spatial correlations $\boldsymbol{K}$ and a diagonal matrix $\phi_{S2S,0} \boldsymbol{I}$.
Thus, we assume that the site term can be decomposed into a spatially correlated part, modeled by $\boldsymbol{K}$, and a remaining "noise" part.
One incorporate this model as is into **Stan** -- however, if one wants to make inferences about both the spatially correlated and the reamining site terms, then it makes sense to separate them.
This is similar to modeling the station terms with a latent variable Gaussian process [@Rasmussen2006]
$$
\delta S \sim N(f_{stat}, \phi_{S2S,0}) \\
f_{stat} \sim GP(0, k(\vec{x}_{s}, \vec{x}_s'))
$$
where $\vec{x}_s$ is the vector of station coordinates, and $k(\vec{x}_{s}, \vec{x}_s'$ is the covariance function.
Here, we use the exponential covariance function
$$
k(\vec{x}_{s}, \vec{x}_s') = \theta_{stat}^2 \exp \left[ -\frac{|\vec{x}_{s} - \vec{x}_{s}'|}{\rho_{stat}}\right]
$$
but other covariances could be used (such as Matern or squared exponential).

In the Stan program, we read in an array of station coordinates `vector[2] X_s[NSTAT]`.
We declare the variance $\theta_{stat}$ and length-scale $\rho_{stat}$ as parameters, and assign priors to them.
For the length scale, it makes sense to assign a prior that penalizes both too small and too large values.
For the variance, we choose an exponential distribution, which has strong support at zer and thus penanlizes the additional complexity of the spatial model [@Simpson2017].

In the model block, we calculate the covariance matrix $K$, and calculate its Cholesky factor.
Then, the spatiall correlated laten variable $f_{stat}$ can be calculated as $f_{stat} = \boldsymbol{L} z$, where $z$ is sandard normally distributed.
Note that to avoid numerical instabilities, we add a small jitter `delta = 1e-9` to the diagonal of the covariance matrix.


```
## data {
##   int<lower=1> N; // number of records
##   int<lower=1> NEQ; // number of earthquakes
##   int<lower=1> NSTAT; // number of earthquakes
##   
##   vector[N] M; // magnitudes
##   vector[N] R; // distances
##   vector[N] VS; // Vs30 values
##   vector[N] Y; // ln PGA values
##   
##   int<lower=1,upper=NEQ> idx_eq[N];
##   int<lower=1,upper=NSTAT> idx_stat[N];
## 
##   vector[2] X_s[NSTAT];  // station coordinate for each record
## }
## 
## transformed data {
##   real vref = 400;
##   real delta = 1e-9;
## }
## 
## parameters {
##   real theta1;
##   real theta2;
##   real theta3;
##   real theta4;
##   real theta5;
##   real theta6;
##   real theta7;
##   
##   real<lower=0> h;
##   
##   real<lower=0> phiSS;
##   real<lower=0> tau;
##   real<lower=0> phiS2S_0;
## 
##   real<lower=0> rho_stat; // length scale of spatial correlation for station terms
##   real<lower=0> theta_stat; // standard deviation of spatial correlation for station terms
##   
##   vector[NEQ] deltaB;
##   vector[NSTAT] deltaS;
## 
##   vector[NSTAT] z_stat;
## }
## 
## model {
##   vector[N] mu;
##   vector[NSTAT] f_stat; // latent vaiable for spatial correlation of station terms
## 
##   theta1 ~ normal(0,10);
##   theta2 ~ normal(0,10);
##   theta3 ~ normal(0,10);
##   theta4 ~ normal(0,10);
##   theta5 ~ normal(0,10);
##   theta6 ~ normal(0,10);
##   theta7 ~ normal(0,10);
##   h ~ normal(6,4);
##   
##   phiSS ~ cauchy(0,0.5);
##   phiS2S_0 ~ cauchy(0,0.5);
##   tau ~ cauchy(0,0.5);
## 
##   rho_stat ~ inv_gamma(2.5,0.1);
##   theta_stat ~ exponential(20);
## 
##   z_stat ~ std_normal();
##   
##   deltaB ~ normal(0,tau);
##   deltaS ~ normal(0,phiS2S_0);
## 
##   // latent variable station contributions to GP
##   {
##     matrix[NSTAT,NSTAT] cov_stat;
##     matrix[NSTAT,NSTAT] L_stat;
## 
##     for(i in 1:NSTAT) {
##       for(j in i:NSTAT) {
##         real d_s;
##   
##         d_s = distance(X_s[i],X_s[j]);
##         cov_stat[i,j] = theta_stat^2 * exp(-d_s/rho_stat);
##         cov_stat[j,i] = cov_stat[i,j];
##       }
##       cov_stat[i,i] = cov_stat[i,i] + delta;
##     }
## 
##     L_stat = cholesky_decompose(cov_stat);
##     f_stat = L_stat * z_stat;
##   }
##   
##   for(i in 1:N) {
##     mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]] + f_stat[idx_stat[i]];
##   }
##   Y ~ normal(mu,phiSS);
## }
```
