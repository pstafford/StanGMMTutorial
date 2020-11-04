data {
  int<lower=1> N; // number of records
  int<lower=1> NEQ; // number of earthquakes
  int<lower=1> NSTAT; // number of stations
  
  vector[N] M; // magnitudes
  vector[N] R; // distances
  vector[NSTAT] VS_obs; // Vs30 values
  vector[NSTAT] VS_sd; // standard deviation of log Vs30 values
  vector[N] Y; // ln PGA values
  
  int<lower=1,upper=NEQ> idx_eq[N];
  int<lower=1,upper=NSTAT> idx_stat[N];
}

transformed data {
  real h = 6;
  real vref = 400;

  vector[NSTAT] lnVS_obs = log(VS_obs);
  real mean_lnVS = mean(lnVS_obs);
  real<lower=0> sd_lnVS = sd(lnVS_obs);
}

parameters {
  real theta1;
  real theta2;
  real theta3;
  real theta4;
  real theta5;
  real theta6;
  real theta7;
  
  real<lower=0> phiSS;
  real<lower=0> tau;
  real<lower=0> phiS2S;
  
  vector[NEQ] deltaB;
  vector[NSTAT] deltaS;
  
  vector<lower=0>[NSTAT] lnVS_star;
}

model {
  theta1 ~ normal(0,10);
  theta2 ~ normal(0,10);
  theta3 ~ normal(0,10);
  theta4 ~ normal(0,10);
  theta5 ~ normal(0,10);
  theta6 ~ normal(0,10);
  theta7 ~ normal(0,10);
  
  phiSS ~ cauchy(0,0.5);
  phiS2S ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);
  
  deltaB ~ normal(0,tau);
  deltaS ~ normal(0,phiS2S);

  lnVS_star ~ normal(mean_lnVS,sd_lnVS);
  lnVS_obs ~ normal(lnVS_star,VS_sd);
  
  for(i in 1:N) {
    real mu;
    mu = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * (lnVS_star[idx_stat[i]] - log(vref)) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
    Y[i] ~ normal(mu, phiSS);
  }
}
