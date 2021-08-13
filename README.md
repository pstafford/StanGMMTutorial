# StanGMMTutorial
Tutorial on using Stan for fitting Ground-motion Models (GMM).

This repository hosts a tutorial about how to use the program Stan (<https://mc-stan.org/>) to estimate the parameters of a ground-motion model (GMM).
Stan is a very flexible program for representing probabilistic models that uses Bayesian inference to estimate the parameters via Markov Chain Monte Carlo (MCMC) sampling.


This repository hosts source files related to the tutorial and makes use of the `R` packages `cmdstanR` to call `stan` and other `R` packages for comparing the various fits and preparing the data.
The files `gmm_stan_tutorial` provide a simple tutorial of how o fit a GMM in Stan on a simulated data set, ad how to assess results.
The files `stan_tutorial_gmmlist` consist of a large set of example Stan models that show how can fit e.g. robust regression GMMs, GMMs with heteroscedastic standard deviation, and spatial models.
The `md` files can be directly viewed from this repository.
The `html` files have the same informatio, and can best be viewed in <https://htmlpreview.github.io/>.
