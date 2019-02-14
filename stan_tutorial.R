# load required packages
library(lme4)
library(rstan)
library(brms)
library(rstanarm)
library(bayesplot)

options(mc.cores = parallel::detectCores())


# read data
setwd('/Users/nico/Dropbox/WORK/STAN_Tutorial/')


data <- read.csv('DATA/NGA_West2_Flatfile_RotD50_d050_public_version_subsetASK.csv', header=TRUE)
dim(data)

h <- 6;
vref = 760;
M = data$Earthquake.Magnitude;
M_sq = M^2;
R = data$ClstD..km.;
lnR = log(sqrt(R^2 + h^2));
MlnR = M * log(sqrt(R^2 + h^2));
lnVS = log(data$Vs30..m.s..selected.for.analysis/vref);

EQID = data$EQID;
STATID = data$Station.Sequence.Number;

Y = log(data$PGA..g.);

data_regression =  data.frame(M,M_sq,R,lnR,MlnR,lnVS,Y,EQID);

fit_lmer = lmer(Y ~ 1 + M_sq + lnR + M * lnR + lnVS
                + (1|EQID),data=data_regression);


eq_idx_factor <- factor(data$EQID)
eq_idx <- as.numeric(eq_idx_factor)

full_d <- list(
  N = length(data[,1]),
  NEQ = max(eq_idx),
  idx_eq = eq_idx,
  M = data$Earthquake.Magnitude,
  R = data$ClstD..km.,
  VS = data$Vs30..m.s..selected.for.analysis,
  Y =log(data$PGA..g.)
);

niter = 400;
wp = 200;
nchains = 4;

fit_model1 <- stan('STAN/gmm_model1_vectorized.stan', data = full_d, 
                       iter = niter, chains = nchains, warmup = wp, verbose = FALSE)

print(get_elapsed_time(fit_model1))

### check
check_treedepth(fit_model1)
check_divergences(fit_model1)

### print and race plot
print(fit_model1, pars = c('lp__','theta','phi','tau'))
traceplot(fit_model1,pars = c('lp__','theta','phi','tau'))

fit_summary <- summary(fit_model1)

posterior <- extract(fit_model1)
deltaB_mean <- colMeans(posterior$deltaB)
M_eq <- unique(data.frame(eq_idx,data$Earthquake.Magnitude[eq_idx]))[,2]
par(mfrow = c(1,2))
plot(M_eq,deltaB_mean)
hist(posterior$theta[,2])

help("brm")

fit_brm <- brm(Y ~ 1 + M_sq + lnR + M * lnR + lnVS
               + (1|EQID),data = data_regression)

summary(fit_brm)
