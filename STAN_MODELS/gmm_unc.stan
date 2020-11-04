// use simulated data, only one period
// with constraints on parameters
// using non-centered parameterization
// uncertainty in M

data {
  int<lower=1> N;  // overall number of records
  int<lower=1> NEQ;   // number of earthquakes

  vector[NEQ] Mobs;       // moment magnitude
  vector[NEQ] Munc;       // moment magnitude
  real R[N];       // Joyner-Boore distance

  int<lower=1,upper=NEQ> eq[N];       // earthquake id

  real Y[N];       // log psa for each record
}

transformed data {
  real mean_M;
  real<lower=0> std_M;


  mean_M = mean(Mobs);
  std_M = sd(Mobs);
}


parameters {
  real<lower=0> sigma_rec;
  real<lower=0> sigma_eq;

  real par1;
  real par2;
  real par3;
  real par4;
  real par5;
  real par6;

  vector[NEQ] eqterm_ind;

  vector<lower=0,upper=10>[NEQ] Mtrue;
}

transformed parameters {
  vector[NEQ] eqterm;

  eqterm = sigma_eq * eqterm_ind;
}

model {
  real mu_rec[N];

  sigma_rec ~ cauchy(0,2);
  sigma_eq ~ cauchy(0,2);

  par1 ~ normal(0,100);
  par2 ~ normal(0,100);
  par3 ~ normal(0,100);
  par4 ~ normal(0,100);
  par5 ~ normal(0,100);
  par6 ~ normal(0,100);


  eqterm_ind ~ normal(0,1);

  //Mtrue ~ normal(mean_M,std_M);

  Mobs ~ normal(Mtrue,Munc);

  for(i in 1:N) {
    real mu1;
 
    mu1 = par1 + par2 * Mtrue[eq[i]] + par3 * Mtrue[eq[i]]^2 + (par4 + par5 * Mtrue[eq[i]]) * log(R[i]) + par6 * R[i];
    mu_rec[i] = mu1 + eqterm[eq[i]];
  }
  Y ~ normal(mu_rec,sigma_rec);
}
