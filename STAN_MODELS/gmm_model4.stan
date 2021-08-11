data {
  int<lower=1> N; // number of records
  int<lower=1> NEQ; // number of earthquakes
  int<lower=1> NSTAT; // number of earthquakes
  int<lower=1> NREG; // number of regions
  
  vector[N] M; // magnitudes
  vector[N] R; // distances
  vector[N] VS; // Vs30 values
  vector[N] Y; // ln PGA values
  
  int<lower=1,upper=NEQ> idx_eq[N];
  int<lower=1,upper=NSTAT> idx_stat[N];
  int<lower=1,upper=NREG> idx_reg[N];
}

transformed data {
  real vref = 400;
}

parameters {
  real mu_theta1;
  real theta2;
  real theta3;
  real theta4;
  real theta5;
  real mu_theta6;
  real mu_theta7;
  
  real<lower=0> h;
  
  real<lower=0> phiSS;
  real<lower=0> tau;
  real<lower=0> phiS2S;
  
  real<lower=0> sigma_theta1;
  real<lower=0> sigma_theta6;
  real<lower=0> sigma_theta7;
  
  vector[NEQ] deltaB;
  vector[NSTAT] deltaS;
  
  vector[NREG] theta1;
  vector<upper=0>[NREG] theta6;
  vector[NREG] theta7;
}

model {
  vector[N] mu;

  mu_theta1 ~ normal(0,10);
  theta2 ~ normal(0,10);
  theta3 ~ normal(0,10);
  theta4 ~ normal(0,10);
  theta5 ~ normal(0,10);
  mu_theta6 ~ normal(0,10);
  mu_theta7 ~ normal(0,10);
  h ~ normal(6,4);
  
  phiSS ~ cauchy(0,0.5);
  phiS2S ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);
  
  sigma_theta1 ~ cauchy(0,0.5);
  sigma_theta6 ~ cauchy(0,0.01);
  sigma_theta7 ~ cauchy(0,0.3);
  
  deltaB ~ normal(0,tau);
  deltaS ~ normal(0,phiS2S);
  
  theta1 ~ normal(mu_theta1,sigma_theta1);
  theta6 ~ normal(mu_theta6,sigma_theta6);
  theta7 ~ normal(mu_theta7,sigma_theta7);
  
  for(i in 1:N) {
    mu[i] = theta1[idx_reg[i]] + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6[idx_reg[i]] * R[i] + theta7[idx_reg[i]] * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
  }
  Y ~ normal(mu,phiSS);
}
