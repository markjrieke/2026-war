functions {
    #include utils.stan
    #include distributions.stan
    #include generated_quantities.stan
}

data {
    // Data dimensions
    int N;                              // Number of observations
    int E;                              // Number of election cycles
    int C;                              // Number of candidates
    int F;                              // Number of non-candidate variables

    // Observations
    matrix[N,F] X;                      // Design matrix
    vector[N] Y;                        // Democratic candidate two-party vote

    // Mapping columns
    array[N,2] int cid;                 // Map candidates to races
    array[N] int eid;                   // Map election cyle to race
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

    // Create counterfactual matrices that eschew incumbency
    matrix[N,F] Xc_dem = standardize_cf(X, iid[1]);
    matrix[N,F] Xc_rep = standardize_cf(X, iid[2]);

    // Number of random walk steps
    int Em = E - 1;

    // Estimate model on the logit scale
    vector[N] Y_logit = logit(Y);
}

parameters {
    real alpha0;                        // Intercept initial state
    vector[Em] eta_alpha;               // Intercept random walk
    real<lower=0> sigma_alpha;          // Intercept random walk scale
    vector[F] beta_v0;                  // Non-candidate parameter random walk initial state
    matrix[F,Em] eta_v;                 // Non-candidate parameter random walk
    vector<lower=0>[F] sigma_v;         // Non-candidate parameter random walk scale
    vector[C] eta_c;                    // Candidate skill parameters
    real<lower=0> sigma_c;              // Candidate skill scale
    vector<lower=0>[E] sigma_e;         // Observation scale
}

transformed parameters {
    // Evaluate hierarchical parameters
    vector[C] beta_c = eta_c * sigma_c;

    // Evaluate the random walks over the intercept and parameters
    vector[E] alpha = random_walk(alpha0, eta_alpha, sigma_alpha);
    matrix[F,E] beta_v = random_walk(beta_v0, eta_v, sigma_v);

    // Estimate the expected mean, sd
    vector[N] mu = latent_mean(Xc, alpha, beta_v, beta_c, eid, cid);
    vector[N] sigma = latent_sd(sigma_e, eid);
}

model {
    // Priors
    target += normal_lpdf(alpha0 | alpha0_mu, alpha0_sigma);
    target += std_normal_lpdf(eta_alpha);
    target += half_normal_lpdf(sigma_alpha | sigma_alpha_sigma);
    target += normal_lpdf(beta_v0 | beta_v0_mu, beta_v0_sigma);
    target += std_normal_lpdf(to_vector(eta_v));
    target += half_normal_lpdf(sigma_v | sigma_v_sigma);
    target += std_normal_lpdf(eta_c);
    target += half_normal_lpdf(sigma_c | sigma_c_sigma);
    target += half_normal_lpdf(sigma_e | sigma_e_sigma);

    // Likelihood
    if (!prior_check) {
        target += normal_lpdf(Y_logit | mu, sigma);
    }
}

generated quantities {
    // Posterior predictive distributions
    vector[N] Y_rep = posterior_predictive_rng(mu, sigma);
    array[2] vector[N] Y_rep_cf = posterior_predictive_cf_rng(
        Xc_dem, Xc_rep, alpha, beta_v, beta_c, sigma_c, sigma, eid, cid
    );

    // Posterior predictive probability of winning
    array[2] vector[N] P_win = win_probability(Y_rep);
    array[2] vector[N] P_win_cf = win_probability(Y_rep_cf);

    // Estimate WAR
    array[2] vector[N] WAR = calculate_WAR(Y_rep, Y_rep_cf);
}