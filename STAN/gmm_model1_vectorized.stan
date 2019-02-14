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
  
  matrix[N,7] X;
  
  for(i in 1:N) {
    X[i,1] = 1;
    X[i,2] = M[i];
    X[i,3] = square(M[i]);
    X[i,4] = log(R[i] + h);
    X[i,5] = M[i] * log(R[i] + h);
    X[i,6] = R[i];
    X[i,7] = log(VS[i]/vref);
  }
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
  
  Y ~ normal(X * theta + deltaB[idx_eq],phi);
}
