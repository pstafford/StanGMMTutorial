  data {
    int<lower=1> N;      // number of records
    int<lower=1> N_eq;   // number of earthquakes
    int<lower=1> N_stat; // number of stations
    int<lower=1> K;      // number of predictors
    
    matrix[N, K] X;      // matrix of predictors
    vector[N] Y;
    
    int<lower=1,upper=N_eq> idx_eq[N];     // event index for each record
    int<lower=1,upper=N_stat> idx_stat[N]; // station index for each record
  }
  
  parameters {
    vector[K] c;             // coefficients
    
    real<lower=0> phi_SS;    // standard deviation for within-event residuals
    real<lower=0> tau;       // standard deviation of site-to-site residuals
    
    vector[N_eq] deltaB;      // event terms
  }
  
  model {
    // prior distributions
    c ~ normal(0,10);
    phi_SS ~ normal(0,1);
    tau ~ normal(0,1);
    
    deltaB ~ normal(0,tau);
    
    Y ~ normal(X * c + deltaB[idx_eq], phi_SS);
  }

  generated quantities {
    vector[N] log_lik;
    for(i in 1:N)
      log_lik[i] = normal_lpdf(Y[i] | X[i] * c + deltaB[idx_eq[i]], phi_SS);
  }
