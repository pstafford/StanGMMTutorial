data {
  int<lower=1> N; // number of records
  int<lower=1> NEQ; // number of earthquakes
  int<lower=1> NSTAT; // number of earthquakes
  
  vector[NEQ] M; // magnitudes
  vector[N] R; // distances
  vector[NSTAT] VS; // Vs30 values
  vector[N] Y; // ln PGA values
  
  int<lower=1,upper=NEQ> idx_eq[N];
  int<lower=1,upper=NSTAT> idx_stat[N];
}

transformed data {
  vector[NSTAT] lnVS;
  vector[NSTAT] VS2;

  for(i in 1:NSTAT) {
    lnVS[i] = fmin(log(VS[i] / 1130), 0.);
    VS2[i] = fmin(VS[i], 1130.);
  }
}

parameters {
  real theta1;
  real theta2;
  real theta3;
  real theta4;
  real theta5;
  real theta6;
  real theta7;

  real phi1;
  real<upper=0> phi2;
  real<upper=0> phi3;
  real<lower=0> phi4;
  
  real<lower=0> h;
  
  real<lower=0> phiSS;
  real<lower=0> tau;
  real<lower=0> phiS2S;
  
  vector[NEQ] deltaB;
  vector[NSTAT] deltaS;
}

model {
  vector[N] mu;

  theta1 ~ normal(0,10);
  theta2 ~ normal(0,10);
  theta3 ~ normal(0,10);
  theta4 ~ normal(0,10);
  theta5 ~ normal(0,10);
  theta6 ~ normal(0,10);
  h ~ normal(6,4);

  phi2 ~ normal(0,1);
  phi3 ~ normal(0,1);
  phi4 ~ normal(0,1);
  
  phiSS ~ cauchy(0,0.5);
  phiS2S ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);
  
  deltaB ~ normal(0,tau);
  deltaS ~ normal(0,phiS2S);
  
  for(i in 1:N) {
    real yref = theta1 + theta2 * M[idx_eq[i]] + theta3 * square(8 - M[idx_eq[i]]) + (theta4 + theta5 * M[idx_eq[i]]) * log(R[i] + h) + theta6 * R[i] + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
    real fsite = phi1 * lnVS[idx_stat[i]];
    real fnl = phi2 * (exp(phi3 * (VS2[idx_stat[i]] - 360)) - exp(phi3 * 770)) * log((exp(yref) + phi4) / phi4);
    mu[i] = yref + fsite + fnl;
  }
  Y ~ normal(mu,phiSS);
}
