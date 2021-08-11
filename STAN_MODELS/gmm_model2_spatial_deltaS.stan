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

  vector[2] X_s[NSTAT];  // station coordinate for each record
}

transformed data {
  real vref = 400;
  real delta = 1e-9;
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
  
  real<lower=0> phiSS;
  real<lower=0> tau;
  real<lower=0> phiS2S_0;

  real<lower=0> rho_stat; // length scale of spatial correlation for station terms
  real<lower=0> theta_stat; // standard deviation of spatial correlation for station terms
  
  vector[NEQ] deltaB;
  vector[NSTAT] deltaS;

  vector[NSTAT] z_stat;
}

model {
  vector[N] mu;
  vector[NSTAT] f_stat; // latent vaiable for spatial correlation of station terms

  theta1 ~ normal(0,10);
  theta2 ~ normal(0,10);
  theta3 ~ normal(0,10);
  theta4 ~ normal(0,10);
  theta5 ~ normal(0,10);
  theta6 ~ normal(0,10);
  theta7 ~ normal(0,10);
  h ~ normal(6,4);
  
  phiSS ~ cauchy(0,0.5);
  phiS2S_0 ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);

  rho_stat ~ inv_gamma(2.5,0.1);
  theta_stat ~ exponential(20);

  z_stat ~ std_normal();
  
  deltaB ~ normal(0,tau);
  deltaS ~ normal(0,phiS2S_0);

  // latent variable station contributions to GP
  {
    matrix[NSTAT,NSTAT] cov_stat;
    matrix[NSTAT,NSTAT] L_stat;

    for(i in 1:NSTAT) {
      for(j in i:NSTAT) {
        real d_s;
  
        d_s = distance(X_s[i],X_s[j]);
        cov_stat[i,j] = theta_stat^2 * exp(-d_s/rho_stat);
        cov_stat[j,i] = cov_stat[i,j];
      }
      cov_stat[i,i] = cov_stat[i,i] + delta;
    }

    L_stat = cholesky_decompose(cov_stat);
    f_stat = L_stat * z_stat;
  }
  
  for(i in 1:N) {
    mu[i] = theta1 + theta2 * M[i] + theta3 * square(8 - M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]] + f_stat[idx_stat[i]];
  }
  Y ~ normal(mu,phiSS);
}
