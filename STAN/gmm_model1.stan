data {
  int<lower=1> N; // number of records
  int<lower=1> NEQ; // number of earthquakes
  
  vector[N] M; // magnitudes
  vector[N] R; // distances
  vector[N] VS; // Vs30 values
  vector[N] Y; // ln PGA values
  
  int<lower=1,upper=NEQ> idx_eq[N];
}

transformed data {
  real h = 6;
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
  
  real<lower=0> phi;
  real<lower=0> tau;
  
  vector[NEQ] deltaB;
}

model {
  theta1 ~ normal(0,10);
  theta2 ~ normal(0,10);
  theta3 ~ normal(0,10);
  theta4 ~ normal(0,10);
  theta5 ~ normal(0,10);
  theta6 ~ normal(0,10);
  theta7 ~ normal(0,10);
  
  phi ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);
  
  deltaB ~ normal(0,tau);
  
  for(i in 1:N) {
    real mu;
    mu = theta1 + theta2 * M[i] + theta3 * square(M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]];
    Y[i] ~ normal(mu,phi);
  }
}
