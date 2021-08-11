data {
  int<lower=1> N;  // overall number of records
  int<lower=1> NEQ;   // number of earthquakes
  int<lower=1> NSTAT; // number of stations
  int<lower=1> NP;

  vector[N] M;       // distance
  vector[N] R;       // distance
  vector[NP] Y[N];       // log psa

  real triggerlevel;

  int<lower=1,upper=NEQ> idx_eq[N];       // earthquake id
  int<lower=1,upper=NSTAT> idx_stat[N];     // station id
}

transformed data {
  vector[NP] mu = rep_vector(0,NP);
  vector[N] M2 = square(8 - M);
  vector[N] lnR = log(R + 6);
  vector[N] MlnR = M .* log(R + 6);
}


parameters {
  vector<lower=0>[NP] phiSS;
  vector<lower=0>[NP] tau;
  vector<lower=0>[NP] phiS2S;

  vector[NP] theta1;
  vector[NP] theta2;
  vector[NP] theta3;
  vector[NP] theta4;
  vector[NP] theta5;
  vector<upper=0>[NP] theta6;

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

  phiSS ~ normal(0,1);
  tau ~ normal(0,1);
  phiS2S ~ normal(0,1);

  theta1 ~ normal(0,5);
  theta2 ~ normal(0,5);
  theta3 ~ normal(0,5);
  theta4 ~ normal(0,5);
  theta5 ~ normal(0,5);
  theta6 ~ normal(0,0.01);

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
      mu_rec[i,p] = theta1[p] + theta2[p] * M[i] +  theta3[p] * M2[i] + theta4[p] * lnR[i] + theta5[p] * MlnR[i] + theta6[p] * R[i] + deltaB[idx_eq[i],p] + deltaS[idx_stat[i],p];
    }
  }

  target += multi_normal_cholesky_lpdf(Y | mu_rec, L_Sigma) - normal_lccdf(triggerlevel | mu_rec[:,1], phiSS[1]);
}
