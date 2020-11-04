data {
  int<lower=1> N; // number of records
  int<lower=1> NEQ; // number of earthquakes
  int<lower=1> NSTAT; // number of earthquakes
  int<lower=1> NP; //  number of periods
  
  vector[N] M; // magnitudes
  vector[N] R; // distances
  vector[N] VS; // Vs30 values
  vector[NP] Y[N];       // log psa
  
  int<lower=1,upper=NEQ> idx_eq[N];
  int<lower=1,upper=NSTAT> idx_stat[N];
}

transformed data {
  real vref = 400;
  vector[NP] mu = rep_vector(0,NP);
}

parameters {
  vector[NP] theta1;
  vector[NP] theta2;
  vector[NP] theta3;
  vector[NP] theta4;
  vector[NP] theta5;
  vector[NP] theta6;
  vector[NP] theta7;
  
  vector<lower=0>[NP] h;
  
  vector<lower=0>[NP] phiSS;
  vector<lower=0>[NP] tau;
  vector<lower=0>[NP] phiS2S;

  vector[NP] deltaB[NEQ];
  vector[NP] deltaS[NSTAT];

  cholesky_factor_corr[NP] L_p;
  cholesky_factor_corr[NP] L_eq;
  cholesky_factor_corr[NP] L_stat;
}

model {
  vector[NP] mu_rec[N];
  matrix[NP,NP] L_Sigma;
  matrix[NP,NP] L_Sigma_eq;
  matrix[NP,NP] L_Sigma_stat;


  theta1 ~ normal(0,10);
  theta2 ~ normal(0,10);
  theta3 ~ normal(0,10);
  theta4 ~ normal(0,10);
  theta5 ~ normal(0,10);
  theta6 ~ normal(0,10);
  theta7 ~ normal(0,10);
  h ~ normal(6,4);
  
  phiSS ~ cauchy(0,0.5);
  phiS2S ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);
  
  L_p ~ lkj_corr_cholesky(1);
  L_Sigma = diag_pre_multiply(phiSS, L_p);

  L_eq ~ lkj_corr_cholesky(1);
  L_Sigma_eq = diag_pre_multiply(tau, L_eq);

  L_stat ~ lkj_corr_cholesky(1);
  L_Sigma_stat = diag_pre_multiply(phiS2S, L_stat);

  deltaB ~ multi_normal_cholesky(mu,L_Sigma_eq);
  deltaS ~ multi_normal_cholesky(mu,L_Sigma_stat);
 
  for(p in 1:NP) {
    for(i in 1:N) {
      mu_rec[i,p] = theta1[p] + theta2[p] * M[i] + theta3[p] * square(8 - M[i]) + (theta4[p] + theta5[p] * M[i]) * log(R[i] + h[p]) + theta6[p] * R[i] + theta7[p] * log(VS[i]/vref) + deltaB[idx_eq[i],p] + deltaS[idx_stat[i],p];
    }
  }
  Y ~ multi_normal_cholesky(mu_rec,L_Sigma);
}
