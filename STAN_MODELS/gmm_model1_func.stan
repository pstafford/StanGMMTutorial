functions {
  real f_median(vector theta, real M, real R, real VS, real h, real vref) {
    real mu = theta[1] + theta[2] * M + theta[3] * square(8 - M) + (theta[4] + theta[5] * M) * log(R + h) + theta[6] * R + theta[7] * log(VS/vref);
    return mu;
  }
}

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
  real vref = 400;
}

parameters {
  vector[7] theta;
  
  real<lower=0> phi;
  real<lower=0> tau;
  
  vector[NEQ] deltaB;
}

model {
  theta ~ normal(0,10);
  
  phi ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);
  
  deltaB ~ normal(0,tau);
  
  for(i in 1:N) {
    real mu;
    mu = f_median(theta, M[i], R[i], VS[i], h, vref) + deltaB[idx_eq[i]];
    Y[i] ~ normal(mu,phi);
  }
}
