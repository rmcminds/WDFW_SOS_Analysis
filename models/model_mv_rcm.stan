data{
  int run_estimation;
  int T;
  int T_forward;
  int T_backward;
  int P;
  int n; 
  array[n] int N_obs;
  int pop_obs[n];
  int year_obs[n];
  vector<lower=0>[P] N_0_med_prior;
  array[P] int interval_start;
}
parameters{
  real<lower=0> tot_var;
  simplex[5] var_props;
  vector<lower=0,upper=1>[4] nu_prop;
  real slope_mu;
  vector[P] eps_slope;
  vector[T-1] eps_annual;
  vector<lower=0>[2] het; 
  vector<lower=0>[P] eps_sigma_rn; 
  row_vector<lower=0>[T-1] eps_sigma_tn; 
  cholesky_factor_corr[P] L;
  matrix[T-1,P] eps;
  row_vector<lower=0>[P] N_0;
  vector[n] overdisp; // allow for extra-poisson 'observation error'
}
transformed parameters{
  vector<lower=0>[P] sigma_rn = sqrt(tot_var * var_props[1]) * eps_sigma_rn^sqrt(0.1*het[1]); // process model std dev per pop
  vector[P] slope = sqrt(tot_var) * (sqrt(var_props[2]) * slope_mu + sqrt(var_props[3]) * eps_slope);
  vector[T-1] annual = sqrt(tot_var * var_props[4]) * eps_annual;
  vector[4] nu = 2.0 / nu_prop; // student dfs modeled with variance paritioning (eg effect of smaller df is additive var); this implies uniform simplex of the variance due to scale vs the variance due to dfs
  matrix<lower=0>[T,P] N;

  N[1,] = N_0;
  for(t in 2:T){
    N[t,] = N[t-1,] .* exp(slope + annual[t-1] + eps_sigma_tn[t-1]^sqrt(0.1*het[2]) * sigma_rn .* (L * eps[t-1,]'))'; // 
  }
}
model{
  vector[n] local_N;
  vector[P] start_prior;
  for(i in 1:n){
    local_N[i] = N[year_obs[i],pop_obs[i]];
  }
  for(p in 1:P) {
    start_prior[p] = log(N_0_med_prior[p]) - interval_start[p] * slope[p] - sum(annual[1:interval_start[p]]);
  } // somewhat redundant back-calculation; only leaves out stochastic process variance and observation error
  
  // =========Priors================
  // mean total variance
  tot_var ~ exponential(1);
  
  // keep variances from extremes
  var_props ~ dirichlet(rep_vector(2,5));
  
  // keep nu from extremes
  nu_prop ~ beta(2,2);

  // slope
  slope_mu ~ student_t(nu[1], 0, sqrt((nu[1]-2)/nu[1])); // 
  eps_slope ~ student_t(nu[2], 0, sqrt((nu[2]-2)/nu[2])); // adjust scale to keep variance == 1
  
  // esu-wide annual effect
  eps_annual ~ student_t(nu[3], 0, sqrt((nu[3]-2)/nu[3]));
  
  // process variance heteroscedasticity (variance of variance scalers)
  het ~ exponential(1);
  
  // process variance sds
  eps_sigma_rn ~ lognormal(0,1);
  eps_sigma_tn ~ lognormal(0,1);

  // correlation matrix
  L ~ lkj_corr_cholesky(1);
  
  // process error
  to_vector(eps) ~ std_normal(); // raw params; do not correspond to specific pops
  
  // initial states
  N_0 ~ lognormal(start_prior, 2 * sqrt(to_vector(interval_start)+1)); // what can we do to remove this? naive removal causes divergences; is there a way to easily marginalize them? maybe simply using better inits would do it?
  
  // observation error
  overdisp ~ student_t(nu[4], log(local_N), sqrt(tot_var * var_props[5] * (nu[4]-2)/nu[4]));
  
  //=========likelihood=============
  if(run_estimation==1){
    N_obs ~ poisson_log(overdisp);
  }
}
generated quantities{
  matrix[P,P] Omega = multiply_lower_tri_self_transpose(L);
  matrix[P,P] Sigma = quad_form_diag(Omega, sigma_rn);
  vector[n] N_sim;
  matrix[T + T_forward + T_backward,P] N_all;
  matrix[T_backward + T + T_forward,P] eps_all;
  N_all[T_backward + 1:T_backward + T,] = N;
  eps_all[T_backward + 1,] = rep_row_vector(0,P);
  eps_all[T_backward + 2:T_backward + T,] = eps;
  if(run_estimation==1){
    for(i in 1:n){
      N_sim[i] = 0;
    }
  }
  if(run_estimation==0){
    for(i in 1:n){
      N_sim[i] = poisson_rng(N[year_obs[i], pop_obs[i]]);
    }
  }
  for(t in (T_backward + T + 1):(T_backward + T + T_forward)){
    for(p in 1:P){
      eps_all[t,p] = normal_rng(0,1);
    }
    N_all[t,] = N_all[t-1,] .* exp(slope + sigma_rn .* (L * eps_all[t,]'))';
  }
  for(t in 1:T_backward){
    for(p in 1:P){
      eps_all[t,p] = normal_rng(0,1);
    }
    N_all[T_backward - t + 1,] = N_all[T_backward - t + 2,] ./ exp(slope - sigma_rn .* (L * eps_all[t,]'))';
  }
}
