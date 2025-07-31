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
    int D;                              // Number of time-varying variables
    int L;                              // Number of time-invariant variables
    int J;                              // Number of sd-estimating variables

    // Observations
    matrix[N,D] Xd;                     // Design matrix (time-varying)
    matrix[N,L] Xl;                     // Design matrix (time-invariant)
    matrix[N,J] Xj;                     // Design matrix (observation sd)
    vector[N] Y;                        // Democratic candidate two-party vote

    // FEC Observations
    array[E] int Sf;                    // Number of campaigns with FEC filings available
    array[E] int Kf;                    // Total number of campaigns per cycle
    // vector[N] Yf;                       // Logit-democratic share of FEC contributions

    // Counterfactual Observations
    matrix[M,D] Xfd;                    // Full matrix (time-varying)
    matrix[M,L] Xfl;                    // Full matrix (time-invariant)
    matrix[M,J] Xfj;                    // Full matrix (observation sd)

    // Mapping columns
    array[N,2] int cid;                 // Map candidates to races in the model frame
    array[N] int eid;                   // Map election cyle to race in the model frame

    // Flag columns for working with counterfactuals
    array[2] int iid;                   // Denote columns that flag incumbency
    // int fid;                            // Column containing FEC contribution share

    // Counterfactual mapping columns
    array[M,2] int cfid;                // Map candidates to races in the full frame
    array[M] int efid;                  // Map election cycle to race in the full frame

    // Priors
    real<lower=0> sigma_sigma;          // Global hyperparameter

    int<lower=0, upper=1> prior_check;
}

transformed data {
    // Center/scale the design time-varying/time-invariant matrices
    matrix[N,D] Xdc = standardize(Xd);
    matrix[N,L] Xlc = standardize(Xl);
    matrix[N,J] Xjc = standardize(Xj);

    // Center/scale counterfactual time-varying/time-invariant matrices based on the design matrices
    matrix[M,D] Xfdc = standardize(Xfd, Xd);
    matrix[M,L] Xflc = standardize(Xfl, Xl);
    matrix[M,J] Xfjc = standardize(Xfj, Xj);

    // Add intercept to national matrices
    int G = L + 1;
    matrix[N,G] Xgc = add_intercept(Xlc);
    matrix[M,G] Xfgc = add_intercept(Xflc);

    // Create counterfactual district matrices that eschew incumbency
    matrix[M,D] Xfdc_dem = standardize_cf(Xfd, Xd, iid[1]);
    matrix[M,D] Xfdc_rep = standardize_cf(Xfd, Xd, iid[2]);

    // Estimate model on the logit scale
    vector[N] Y_logit = logit(Y);
}

parameters {
    // Global hyper-parameter
    real<lower=0> sigma;

    // Variable scale parameter offsets
    real<lower=0> eta_sigma_alpha;
    vector<lower=0>[D] eta_sigma_beta_d;
    real<lower=0> eta_sigma_beta_g;
    real<lower=0> eta_sigma_beta_j;
    real<lower=0> eta_sigma_beta_c;
    real<lower=0> eta_sigma_sigma_e;
    real<lower=0> eta_sigma_beta_f;

    // Variable offsets
    vector[E] eta_alpha;
    matrix[D,E] eta_beta_d;
    vector[G] eta_beta_g;
    vector[J] eta_beta_j;
    vector[C] eta_beta_c;
    vector[E] eta_sigma_e;
    vector[E] eta_beta_f;
}

transformed parameters {
    // Evaluate hierarchical parameters
    vector[C] beta_c = eta_beta_c * eta_sigma_beta_c * sigma;
    vector[G] beta_g = eta_beta_g * sigma;
    vector[J] beta_j = eta_beta_j * sigma;

    // Conditionally use additional hierarchical parameter if necessary
    if (G > 1) {
        beta_g *= eta_sigma_beta_g;
    }

    if (J > 1) {
        beta_j *= eta_sigma_beta_j;
    }

    // Evaluate random walk over the intercept
    vector[E] alpha = eta_alpha * eta_sigma_alpha * sigma;
    for (i in 2:E) {
        alpha[i] += alpha[i-1];
    }

    // Evaluate random walk over the parameters
    matrix[D,E] beta_d = diag_pre_multiply(eta_sigma_beta_d * sigma, eta_beta_d);
    for (i in 2:E) {
        beta_d[:,i] += beta_d[:,i-1];
    }

    // Evaluate random walk over the standard deviations
    vector[E] sigma_e = eta_sigma_e * eta_sigma_sigma_e * sigma;
    for (i in 2:E) {
        sigma_e[i] += sigma_e[i-1];
    }

    // Evaluate random walk over FEC filing probability
    vector[E] beta_f = eta_beta_f * eta_sigma_beta_f * sigma;
    for (i in 2:E) {
        beta_f[i] += beta_f[i-1];
    }
}

model {
    // Estimate the expected mean, sd
    vector[N] mu = latent_mean(Xdc, Xgc, alpha, beta_d, beta_g, beta_c, eid, cid);
    vector[N] sigma_o = latent_sd(Xjc, beta_j, sigma_e, eid);

    // Global hyper-prior
    target += half_normal_lpdf(sigma | sigma_sigma);

    // Variable scale priors
    target += std_half_normal_lpdf(eta_sigma_alpha);
    target += std_half_normal_lpdf(eta_sigma_beta_d);
    target += std_half_normal_lpdf(eta_sigma_beta_g);
    target += std_half_normal_lpdf(eta_sigma_beta_j);
    target += std_half_normal_lpdf(eta_sigma_beta_c);
    target += std_half_normal_lpdf(eta_sigma_sigma_e);
    target += std_half_normal_lpdf(eta_sigma_beta_f);

    // Variable offsets
    target += std_normal_lpdf(eta_alpha);
    target += std_normal_lpdf(to_vector(eta_beta_d));
    target += std_normal_lpdf(eta_beta_g);
    target += std_normal_lpdf(eta_beta_j);
    target += std_normal_lpdf(eta_beta_c);
    target += std_normal_lpdf(eta_sigma_e);
    target += std_normal_lpdf(to_vector(eta_beta_f));

    // Likelihood
    if (!prior_check) {
        target += normal_lpdf(Y_logit | mu, sigma_o);
        target += binomial_logit_lpmf(Sf | Kf, beta_f);
    }
}

generated quantities {
    // Posterior predictive distributions
    vector[M] Y_rep = posterior_predictive_rng(
        Xfdc,
        Xfgc,
        Xfjc,
        alpha,
        beta_d,
        beta_g,
        beta_j,
        beta_c,
        sigma_e,
        efid,
        cfid
    );

    // Counterfactual predictive distributions
    array[2] vector[M] Y_rep_cf = posterior_predictive_cf_rng(
        Xfdc_dem,
        Xfdc_rep,
        Xfgc,
        Xfjc,
        alpha,
        beta_d,
        beta_g,
        beta_j,
        beta_c,
        eta_sigma_beta_c * sigma,
        sigma_e,
        efid,
        cfid
    );

    // Posterior predictive probability of winning
    array[2] vector[M] P_win = win_probability(Y_rep);
    array[2] vector[M] P_win_cf = win_probability(Y_rep_cf);

    // Estimate WAR
    array[2] vector[M] WAR = calculate_WAR(Y_rep, Y_rep_cf);
}