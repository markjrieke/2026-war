functions {
    #include utils.stan
    #include distributions.stan
    #include generated_quantities.stan
}

data {
    // Data dimensions
    int N;                              // Number of observations in the model frame
    int M;                              // Number of observations in the full frame
    int E;                              // Number of election cycles
    int C;                              // Number of candidates
    int F;                              // Number of non-candidate variables

    // Observations
    matrix[N,F] X;                      // Design matrix
    vector[N] Y;                        // Democratic candidate two-party vote
    matrix[M,F] Xf;                     // Full matrix

    // Mapping columns
    array[N,2] int cid;                 // Map candidates to races in the model frame
    array[N] int eid;                   // Map election cyle to race in the model frame
    array[M,2] int cfid;                // Map candidates to races in the full frame
    array[M] int efid;                  // Map election cycle to race in the full frame
    array[2] int iid;                   // Denote columns that flag incumbency

    // Priors
    real alpha0_mu;                     // Intercept prior mean
    real<lower=0> alpha0_sigma;         // Intercept prior scale
    real<lower=0> sigma_alpha_sigma;    // Intercept random walk scale
    real beta_v0_mu;                    // Non-candidate parameter initial state prior mean
    real beta_v0_sigma;                 // Non-candidate parameter initial state prior scale
    real<lower=0> sigma_v_sigma;        // Non-candidate parameter random walk scale
    real<lower=0> sigma_c_sigma;        // Candidate effect scale prior scale
    real<lower=0> sigma_e_sigma;        // Observation prior scale

    int<lower=0, upper=1> prior_check;
}

transformed data {
    // Center/scale the design matrix
    matrix[N,F] Xc = standardize(X);
    matrix[M,F] Xfc = standardize(Xf);

    // Create counterfactual matrices that eschew incumbency
    matrix[M,F] Xc_dem = standardize_cf(Xf, iid[1]);
    matrix[M,F] Xc_rep = standardize_cf(Xf, iid[2]);

    // Number of random walk steps
    int Em = E - 1;

    // Estimate model on the logit scale
    vector[N] Y_logit = logit(Y);
}

parameters {
    real<lower=0> sigma;
    real<lower=0> eta_sigma_alpha;
    vector[E] eta_alpha;
    vector<lower=0>[F] eta_sigma_beta_v;
    matrix[F,E] eta_beta_v;
    real<lower=0> eta_sigma_beta_c;
    vector[C] eta_beta_c;
    vector<lower=0>[E] eta_sigma_e;
}

transformed parameters {
    // Evaluate hierarchical parameters
    vector[C] beta_c = eta_beta_c * eta_sigma_beta_c * sigma;

    // Evaluate random walk over the intercept
    vector[E] alpha = eta_alpha * eta_sigma_alpha * sigma;
    for (i in 2:E) {
        alpha[i] += alpha[i-1];
    }

    // Evaluate random walk over the parameters
    matrix[F,E] beta_v = diag_pre_multiply(eta_sigma_beta_v * sigma, eta_beta_v);
    for (i in 2:E) {
        beta_v[:,i] += beta_v[:,i-1];
    }

    // Evaluate the random walks over the intercept and parameters
    // vector[E] alpha = random_walk(alpha0, eta_alpha, sigma_alpha);
    // matrix[F,E] beta_v = random_walk(beta_v0, eta_v, sigma_v);
}

model {
    // Estimate the expected mean, sd
    vector[N] mu = latent_mean(Xc, alpha, beta_v, beta_c, eid, cid);
    vector[N] sigma_o = latent_sd(eta_sigma_e * sigma, eid);

    // Priors
    // target += normal_lpdf(alpha0 | alpha0_mu, alpha0_sigma);
    // target += std_normal_lpdf(eta_alpha);
    // target += half_normal_lpdf(sigma_alpha | sigma_alpha_sigma);
    // target += normal_lpdf(beta_v0 | beta_v0_mu, beta_v0_sigma);
    // target += std_normal_lpdf(to_vector(eta_v));
    // target += half_normal_lpdf(sigma_v | sigma_v_sigma);
    // target += std_normal_lpdf(eta_c);
    // target += half_normal_lpdf(sigma_c | sigma_c_sigma);
    // target += half_normal_lpdf(sigma_e | sigma_e_sigma);
    target += half_normal_lpdf(sigma | 1);
    target += half_normal_lpdf(eta_sigma_alpha | 1);
    target += std_normal_lpdf(eta_alpha);
    target += half_normal_lpdf(eta_sigma_beta_v | 1);
    target += std_normal_lpdf(to_vector(eta_beta_v));
    target += half_normal_lpdf(eta_sigma_beta_c | 1);
    target += std_normal_lpdf(eta_beta_c);
    target += half_normal_lpdf(eta_sigma_e | 1);

    // Likelihood
    if (!prior_check) {
        target += normal_lpdf(Y_logit | mu, sigma_o);
    }
}

generated quantities {
    // Posterior predictive distributions
    vector[M] Y_rep = posterior_predictive_rng(
        Xfc, alpha, beta_v, beta_c, eta_sigma_e * sigma, efid, cfid
    );

    // Counterfactual predictive distributions
    array[2] vector[M] Y_rep_cf = posterior_predictive_cf_rng(
        Xc_dem, Xc_rep, alpha, beta_v, beta_c, eta_sigma_beta_c * sigma, eta_sigma_e * sigma, efid, cfid
    );

    // Posterior predictive probability of winning
    array[2] vector[M] P_win = win_probability(Y_rep);
    array[2] vector[M] P_win_cf = win_probability(Y_rep_cf);

    // Estimate WAR
    array[2] vector[M] WAR = calculate_WAR(Y_rep, Y_rep_cf);
}