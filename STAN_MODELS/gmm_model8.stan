data {
  int<lower=1> N; // number of records
  int<lower=1> NEQ; // number of earthquakes
  int<lower=1> NSTAT; // number of earthquakes
  
  vector[N] M; // magnitudes
  vector[N] R; // distances
  vector[N] VS; // Vs30 values
  vector[N] Y; // ln PGA values
  
  int<lower=1,upper=NEQ> idx_eq[N];
  int<lower=1,upper=NSTAT> idx_stat[N];
}

transformed data {
  real vref = 760;
}

parameters {
  real theta1;
  real theta2;
  real theta3;
  real theta4;
  real theta5;
  real theta6;
  real theta7;
  
  real<lower=0> h;
  
  real<lower=0> phiSS_1;
  real<lower=0> phiSS_2;
  real<lower=0> tau;
  real<lower=0> phiS2S;
  
  vector[NEQ] deltaB;
  vector[NSTAT] deltaS;
  
  simplex[2] wt;
}

model {
  vector[N] mu;

  theta1 ~ normal(0,10);
  theta2 ~ normal(0,10);
  theta3 ~ normal(0,10);
  theta4 ~ normal(0,10);
  theta5 ~ normal(0,10);
  theta6 ~ normal(0,10);
  theta7 ~ normal(0,10);
  h ~ normal(6,4);
  
  phiSS_1 ~ cauchy(0,0.5);
  phiSS_2 ~ cauchy(0,0.5);
  phiS2S ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);
  
  deltaB ~ normal(0,tau);
  deltaS ~ normal(0,phiS2S);
  
  for(i in 1:N) {
    mu[i] = theta1 + theta2 * M[i] + theta3 * square(M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
    
    target += log_sum_exp(log(wt[1]) + normal_lpdf(Y[i] | mu[i], phiSS_1), log(wt[2]) + normal_lpdf(Y[i] | mu[i], phiSS_2));
  }
}
