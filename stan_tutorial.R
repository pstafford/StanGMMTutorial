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

### print and trace plot
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



### check spatial correlation code ----
library(geosphere)

# sort the data by event to obtain a block diagonal structure
sdata <- data[order(EQID), ]

# compute the number of observations per event
uEQID <- unique(sdata$EQID)
numEvents <- length(uEQID)
obsPerEvent <- rep(0, numEvents)
for ( i in 1:numEvents ) {
  obsPerEvent[i] <- length(which(sdata$EQID == uEQID[i]))
}

# get station longitude and latitudes
slon <- sdata$Station.Longitude
slat <- sdata$Station.Latitude

# compute the full inter-station distances
spos <- as.matrix(cbind(slon, slat))

# there is a co-located station within event 1030, so add some noise to one station RecID 11605
rid <- which(sdata$Record.Sequence.Number==11605)
spos[rid,] <- spos[rid,] + rnorm(2,0,1e-4)

dij <- distm(spos, fun = distHaversine)/1e3

# event and station indices
eq_idx_factor <- factor(sdata$EQID)
eq_idx <- as.numeric(eq_idx_factor)

stat_idx_factor <- factor(sdata$Station.Sequence.Number)
stat_idx <- as.numeric(stat_idx_factor)

# input data structures (free and fixed correlation length versions)

full_d_sp_free <- list(
  N = nrow(sdata),
  NEQ = max(eq_idx),
  NSTAT = max(stat_idx),
  M = sdata$Earthquake.Magnitude,
  R = sdata$ClstD..km.,
  VS = sdata$Vs30..m.s..selected.for.analysis,
  Y = log(sdata$PGA..g.),
  idx_eq = eq_idx,
  idx_stat = stat_idx,
  distanceMatrix = dij,
  ObsPerEvent = obsPerEvent
)

# Note that this Jayarm & Baker give:
# 40.7 - 15.0*T for clustered Vs30 regions, and 
#  8.5 + 17.2*T otherwise
# but these are within the context of a model exp(-3*dx/h)
# As we're using a simple model here we should expect apparent clustering from model bias,
# so use a relatively high value (h=30 --> correlationLength=10)
correlationLength <- 10.0

# create a blocked diagonal correlation matrix
oij <- outer(sdata$EQID, sdata$EQID, FUN="/")
oij[oij != 1] <- 0
rhoij <- oij * exp(-dij/correlationLength)

full_d_sp_fixed <- list(
  N = nrow(sdata),
  NEQ = max(eq_idx),
  NSTAT = max(stat_idx),
  M = sdata$Earthquake.Magnitude,
  R = sdata$ClstD..km.,
  VS = sdata$Vs30..m.s..selected.for.analysis,
  Y = log(sdata$PGA..g.),
  idx_eq = eq_idx,
  idx_stat = stat_idx,
  spatialCorrelationMatrix = rhoij,
  ObsPerEvent = obsPerEvent
)


niter = 4;
wp = 2;
nchains = 4;


fit_model1_spatial_free <- stan('STAN/gmm_model1_spatial_free.stan', data = full_d_sp_free, 
                   iter = niter, chains = nchains, warmup = wp, verbose = FALSE)

print(fit_model1_spatial_free, pars = c('lp__','phiSS','tau','phiS2S','correlationLength'))
traceplot(fit_model1_spatial_free, pars = c('lp__','phiSS','tau','phiS2S','correlationLength'))


fit_model1_spatial_fixed <- stan('STAN/gmm_model1_spatial_fixed.stan', data = full_d_sp_fixed, 
                                iter = niter, chains = nchains, warmup = wp, verbose = FALSE)

print(fit_model1_spatial_fixed, pars = c('lp__','phiSS','tau','phiS2S'))
traceplot(fit_model1_spatial_fixed, pars = c('lp__','phiSS','tau','phiS2S'))

