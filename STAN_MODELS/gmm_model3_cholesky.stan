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
  real vref = 400;
  vector[2] mu_eq;
  
  for(i in 1:2)
    mu_eq[i] = 0;
}

parameters {
  real theta1;
  real theta2;
  real theta3;
  real theta4;
  real theta5;
  real theta6;
  real theta7;
  
  real a;
  real b;
  
  real<lower=0> phiSS;
  vector<lower=0>[2] tau;
  real<lower=0> phiS2S;
  
  cholesky_factor_corr[2] L_eq;
  matrix[2,NEQ] z_eq;
  
  vector[NSTAT] deltaS;
}

transformed parameters {
  matrix[2,NEQ] deltaB;

  deltaB = diag_pre_multiply(tau, L_eq) * z_eq;

}

model {
  vector[N] mu;
  matrix[2,2] Sigma_eq;

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
  
  deltaS ~ normal(0,phiS2S);
  
  L_eq ~ lkj_corr_cholesky(2);
  to_vector(z_eq) ~ std_normal();
  
  
  for(i in 1:N) {
    mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + a * exp(b * M[i] + deltaB[2,idx_eq[i]])) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[1,idx_eq[i]] + deltaS[idx_stat[i]];
  }
  Y ~ normal(mu,phiSS);
}
