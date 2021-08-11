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
    real<lower=0> phi_S2S;   // standard deviation of between-event residuals
    real<lower=0> tau;       // standard deviation of site-to-site residuals
    
    vector[N_eq] deltaB;      // event terms
    vector[N_stat] deltaS;    // station terms
  }
  
  model {
    // prior distributions
    c ~ normal(0,10);
    phi_SS ~ normal(0,1);
    tau ~ normal(0,1);
    phi_S2S ~ normal(0,1);
    
    deltaB ~ normal(0,tau);
    deltaS ~ normal(0,phi_S2S);
    
    Y ~ normal(X * c + deltaB[idx_eq] + deltaS[idx_stat], phi_SS);
  }
