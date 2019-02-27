# StanGMMTutorial
Tutorial on using Stan for fitting Ground-motion Models (GMM).

This repository hosts a tutorial about how to use the program Stan (<https://mc-stan.org/>) to estimate the parameters of a ground-motion model (GMM).
Stan is a very flexible program for representing probabilistic models that uses Bayesian inference to estimate the parameters via Markov Chain Monte Carlo (MCMC) sampling.

This repository hosts source files related to the tutorial and makes use of the `R` packages `rstan` to call `stan` and other `R` packages for comparing the various fits and preparing the data.
The main file `gmpe_stan_tutorial.Rmd` can be used within `RStudio` to create stand-alone html or pdf files. 
This same file also contains all relevant code and is a live document.
