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
  
  matrix[N,N] spatialCorrelationMatrix; // within-event correlation matrix (block diagonal matrix)
  int<lower=1,upper=NEQ> ObsPerEvent[NEQ];
}

transformed data {
  real vref = 760;
  // taking Cholesky here is only valid if spatial correlation matrix is block diagonal
  cholesky_factor_corr[N] LspatialCorrelationMatrix = cholesky_decompose(spatialCorrelationMatrix);
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
  theta7 ~ normal(0,10);
  h ~ normal(6,4);
  
  phiSS ~ cauchy(0,0.5);
  phiS2S ~ cauchy(0,0.5);
  tau ~ cauchy(0,0.5);
  
  deltaB ~ normal(0,tau);
  deltaS ~ normal(0,phiS2S);
  
  for(i in 1:N) {
    mu[i] = theta1 + theta2 * M[i] + theta3 * square(M[i]) + (theta4 + theta5 * M[i]) * log(R[i] + h) + theta6 * R[i] + theta7 * log(VS[i]/vref) + deltaB[idx_eq[i]] + deltaS[idx_stat[i]];
  }

  {
    int epos;
    epos = 1;
    for ( i in 1:NEQ ) {
      if ( ObsPerEvent[i] > 1 ) {
        vector[ObsPerEvent[i]] eventMu = segment(mu, epos, ObsPerEvent[i]);
        matrix[ObsPerEvent[i],ObsPerEvent[i]] eventLCov = phiSS * block( LspatialCorrelationMatrix, epos, epos, ObsPerEvent[i], ObsPerEvent[i] );
        segment(Y, epos, ObsPerEvent[i]) ~ multi_normal_cholesky( eventMu, eventLCov );
      } else {
        real eventMu = mu[epos];
        Y[epos] ~ normal( eventMu, phiSS );
      }
      epos = epos + ObsPerEvent[i];
    }
  }
}
