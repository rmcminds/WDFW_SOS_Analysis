data{
  int run_estimation;
  int T;
  int T_forward;
  int T_backward;
  int P;
  int n; 
  vector[n] N_obs;
  int pop_obs[n];
  int year_obs[n];
  vector<lower=0>[P] N_0_med_prior;
}
parameters{
  matrix[T-1,P] eps;
  vector[P] eps_slope;
  real slope_mu;
  real<lower=0> sigma_slope;
  row_vector<lower=0>[P] N_0;
  real<lower=0> sigma_rn_mu;
  real<lower=0> sigma_wn_mu;
  real<lower=0> sigma_rn_sigma;
  real<lower=0> sigma_wn_sigma;
  vector<lower=0>[P] eps_sigma_rn; 
  vector<lower=0>[P] eps_sigma_wn; 
  cholesky_factor_corr[P] L;
}
transformed parameters{
  matrix<lower=0>[T,P] N;
  vector<lower=0>[P] sigma_rn = sigma_rn_mu + eps_sigma_rn * sigma_rn_sigma; // process model std dev per pop
  vector<lower=0>[P] sigma_wn = sigma_wn_mu + eps_sigma_wn * sigma_wn_sigma; // obs model std dev per pop
  
  N[1,] = N_0;
  for(t in 2:T){
    N[t,] = N[t-1,] .* exp(slope_mu + sigma_slope * eps_slope + sigma_rn .* (L * eps[t-1,]'))'; // 
  }
}
model{
  vector[n] local_N;
  vector[n] local_sigma_wn;
  for(i in 1:n){
    local_N[i] = N[year_obs[i],pop_obs[i]];
    local_sigma_wn[i] = sigma_wn[pop_obs[i]];
  }
  
  // =========Priors================
  // slope
  slope_mu ~ normal(0,0.25); 
  sigma_slope ~ cauchy(0,0.1);
  eps_slope[1:P] ~ std_normal();
  
  //observation  & process error sds
  sigma_rn_mu ~ inv_gamma(1,0.125); 
  sigma_wn_mu ~ inv_gamma(1,0.125);
  sigma_rn_sigma ~ cauchy(0,0.1);
  sigma_wn_sigma ~ cauchy(0,0.1);
  eps_sigma_rn ~ cauchy(0,1);
  eps_sigma_wn ~ cauchy(0,1);
  
  //correlation matrix
  L ~ lkj_corr_cholesky(1);
  
  //process errors
  to_vector(eps) ~ std_normal();
  
  //initial states
  N_0 ~ lognormal(log(N_0_med_prior),2);
  
  //=========likelihood=============
  if(run_estimation==1){
    log(N_obs) ~ normal(log(local_N), local_sigma_wn); // add poisson? the 'observation error' seems redundant since there's only one observation per population per year, so the normal distribution for 'process error' has the same degrees of freedom
  }
}
generated quantities{
  vector[P] slope;
  matrix[P,P] Omega = multiply_lower_tri_self_transpose(L);
  matrix[P,P] Sigma = quad_form_diag(Omega, sigma_rn);
  vector[n] N_sim;
  matrix[T + T_forward + T_backward,P] N_all;
  matrix[T_backward + T + T_forward,P] eps_all;
  N_all[T_backward + 1:T_backward + T,1:P] = N;
  eps_all[T_backward + 1,1:P] = rep_row_vector(0,P);
  eps_all[T_backward + 2:T_backward + T,1:P] = eps;
  if(run_estimation==1){
    for(i in 1:n){
      N_sim[i] = 0;
    }
  }
  if(run_estimation==0){
    for(i in 1:n){
      N_sim[i] = lognormal_rng(log(N[year_obs[i],pop_obs[i]]), sigma_wn[pop_obs[i]]);
    }
  }
  for(p in 1:P){
    slope[p] = slope_mu + eps_slope[p] * sigma_slope;
  }
  for(t in (T_backward + T + 1):(T_backward + T + T_forward)){
    for(p in 1:P){
      eps_all[t,p] = normal_rng(0,1);
    }
    N_all[t,1:P] = to_row_vector(exp(to_vector(log(N_all[t-1,1:P])) + slope[1:P] + L * to_vector(eps_all[t,1:P])));
  }
  for(t in 1 : T_backward){
    for(p in 1:P){
      eps_all[t,p] = normal_rng(0,1);
    }
    N_all[T_backward - t + 1,1:P] = to_row_vector(exp(to_vector(log(N_all[T_backward - t + 2,1:P])) - slope[1:P] - L * to_vector(eps_all[t,1:P])));
  }
}
