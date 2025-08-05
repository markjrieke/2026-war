functions {
    #include utils.stan
    #include distributions.stan
    #include generated_quantities.stan
}

data {
    // Data dimensions
    int N;                              // Number of observations in the model frame
    int M;                              // Number of observations in the full frame
    int H;                              // Number of election cycles in the model frame
    int E;                              // Number of election cycles in the full frame
    int C;                              // Number of candidates
    int D;                              // Number of time-varying variables
    int G;                              // Number of time-invariant variables
    int J;                              // Number of sd-estimating variables
    int F;                              // Number of FEC predictors

    // Observations
    matrix[N,D] Xd;                     // Design matrix (time-varying)
    matrix[N,G] Xg;                     // Design matrix (time-invariant)
    matrix[N,J] Xj;                     // Design matrix (observation sd)
    vector[N] Y;                        // Democratic candidate two-party vote

    // FEC Observations
    matrix[N,F] Xf;                     // Design matrix for FEC predictors
    array[H] int Sf;                    // Number of campaigns with FEC filings available
    array[H] int Kf;                    // Total number of campaigns per cycle
    array[E] int fec;                   // Whether (or not) FEC data is available for a given year
    vector[N] Yf;                       // Logit-democratic share of FEC contributions

    // Experience Observations
    array[2,H] int Ke;                  // Number of non-incumbents per party per cycle
    array[2,H] int Ye;                  // Number of experienced non-incumbents per party per cycle

    // Counterfactual Observations
    matrix[M,D] Xfd;                    // Full matrix (time-varying)
    matrix[M,G] Xfg;                    // Full matrix (time-invariant)
    matrix[M,J] Xfj;                    // Full matrix (observation sd)

    // Counterfactual FEC Observations
    matrix[M,F] Xff;                    // Full matrix for FEC predictors
    vector[M] Yff;                      // Logit-democratic share of FEC contributions

    // Counterfactual Experience Observations
    matrix[M,2] Xfe;                    // Full matrix of experience levels

    // Mapping columns
    array[N,2] int cid;                 // Map candidates to races in the model frame
    array[N] int eid;                   // Map election cyle to race in the model frame

    // Flag columns for working with counterfactuals
    array[2] int iid;                   // Denote columns that flag incumbency
    array[2] int exid_f;                // Columns that flag experience advantage/disadvantage in Xf
    array[2] int exid_d;                // Columns that flag experience advantage/disadvantage in Xd
    int fid;                            // Column containing FEC contribution share

    // Counterfactual mapping columns
    array[M,2] int cfid;                // Map candidates to races in the full frame
    array[M] int efid;                  // Map election cycle to race in the full frame

    // Priors
    real<lower=0> sigma_sigma;          // Global hyperparameter

    int<lower=0, upper=1> prior_check;
}

transformed data {
    // Center/scale the design matrices
    matrix[N,D] Xdc = standardize(Xd);
    matrix[N,G] Xgc = standardize(Xg);
    matrix[N,J] Xjc = standardize(Xj);
    matrix[N,F] Xfc = standardize(Xf);

    // Center/scale counterfactual time-varying/time-invariant matrices based on the design matrices
    matrix[M,D] Xfdc = standardize(Xfd, Xd);
    matrix[M,G] Xfgc = standardize(Xfg, Xg);
    matrix[M,J] Xfjc = standardize(Xfj, Xj);
    matrix[M,F] Xffc = standardize(Xff, Xf);

    // Create counterfactual district matrices that eschew incumbency
    // These get further modified in the generated quantities block as a part of
    // generating the counterfactual, but counterfactual incumbency is always 0.
    matrix[M,D] Xfdc_dem = standardize_cf(Xfd, Xd, iid[1]);
    matrix[M,D] Xfdc_rep = standardize_cf(Xfd, Xd, iid[2]);

    // Estimate model on the logit scale
    vector[N] Y_logit = logit(Y);

    // Reference FEC share in terms of %
    vector[M] Yff_inv_logit = inv_logit(Yff);

    // Set FEC 'cases' based on FEC share
    // 1: D candidate has ~ 100% of reported FEC share
    // 2: R candidate has ~ 100% of reported FEC share
    // 3: Neither candidate has reported FEC values
    // 4: Normal mix of reported FEC contributions
    vector[M] cases;
    for (m in 1:M) {
        if (Yff_inv_logit[m] > 0.99) {
            cases[m] = 1;
        } else if (Yff_inv_logit[m] < 0.01) {
            cases[m] = 2;
        } else if (Yff_inv_logit[m] == 0.5) {
            cases[m] = 3;
        } else {
            cases[m] = 4;
        }
    }

    // Centered/scaled experience advantage/disadvantage values
    vector[2] xe_mean;
    vector[2] xe_sd;
    for (p in 1:2) {
        xe_mean[p] = mean(Xf[:,exid_f[p]]);
        xe_sd[p] = sd(Xf[:,exid_f[p]]);
    }

    // Number of random walk steps
    int Em = E - 1;
}

parameters {
    // Global hyper-parameter
    real<lower=0> sigma;

    // Variable scale parameter offsets
    real<lower=0> eta_sigma_alpha;          // Time-varying intercept random walk scale
    vector<lower=0>[D] eta_sigma_beta_d;    // Time-varying parameter random walk scale
    real<lower=0> eta_sigma_beta_g;         // Time-invariant parameter scale
    real<lower=0> eta_sigma_beta_j;         // Standard deviation predictor scale
    real<lower=0> eta_sigma_beta_c;         // Candidate skill scale
    real<lower=0> eta_sigma_sigma_e;        // Standard deviation random walk scale
    real<lower=0> eta_sigma_theta_f;        // FEC hurdle probability random walk scale
    real<lower=0> eta_sigma_alpha_f;        // FEC share intercept random walk scale
    real<lower=0> eta_sigma_beta_f;         // FEC share parameter scale
    real<lower=0> eta_sigma_sigma_f;        // FEC share standard deviation random walk scale
    vector[2] eta_sigma_theta_e;            // Experience probability random walk scale

    // Random walk initial states
    real eta_alpha0;                        // Time-varying intercept initial state
    vector[D] eta_beta_d0;                  // Time varying parameter initial state
    real eta_sigma_e0;                      // Standard deviation initial state
    real eta_theta_f0;                      // FEC hurdle probability initial state
    real eta_alpha_f0;                      // FEC share intercept initial state
    real eta_sigma_f0;                      // FEC share standard deviation initial state
    vector[2] eta_theta_e0;                 // Experience probability initial state

    // Random walk steps
    vector[Em] eta_alpha;                   // Time-varying intercept random walk steps
    matrix[D,Em] eta_beta_d;                // Time-varying parameter random walk steps
    vector[G] eta_beta_g;                   // Time-invariant parameters
    vector[J] eta_beta_j;                   // Standard deviation predictors
    vector[C] eta_beta_c;                   // Candidate skill
    vector[Em] eta_sigma_e;                 // Standard deviation random walk steps
    vector[Em] eta_theta_f;                 // FEC hurdle probability random walk steps
    vector[Em] eta_alpha_f;                 // FEC share intercept random walk steps
    vector[F] eta_beta_f;                   // FEC share parameters
    vector[Em] eta_sigma_f;                 // FEC share standard deviation random walk steps
    matrix[2,Em] eta_theta_e;               // Experience probability random walk steps
}

transformed parameters {
    // Evaluate hierarchical parameters
    vector[C] beta_c = evaluate_hierarchy(eta_beta_c, eta_sigma_beta_c, sigma);
    vector[G] beta_g = evaluate_hierarchy(eta_beta_g, eta_sigma_beta_g, sigma);
    vector[J] beta_j = evaluate_hierarchy(eta_beta_j, eta_sigma_beta_j, sigma);
    vector[F] beta_f = evaluate_hierarchy(eta_beta_f, eta_sigma_beta_f, sigma);

    // Evaluate random walks over vectors
    vector[E] alpha = random_walk(eta_alpha0, eta_alpha, eta_sigma_alpha, sigma);
    vector[E] sigma_e = random_walk(eta_sigma_e0, eta_sigma_e, eta_sigma_sigma_e, sigma);
    vector[E] theta_f = random_walk(eta_theta_f0, eta_theta_f, eta_sigma_theta_f, sigma);
    vector[E] alpha_f = random_walk(eta_alpha_f0, eta_alpha_f, eta_sigma_alpha_f, sigma);
    vector[E] sigma_f = exp(random_walk(eta_sigma_f0, eta_sigma_f, eta_sigma_sigma_f, sigma));

    // Evaluate random walk over matrices
    matrix[D,E] beta_d = random_walk(eta_beta_d0, eta_beta_d, eta_sigma_beta_d, sigma);
    matrix[2,E] theta_e = random_walk(eta_theta_e0, eta_theta_e, eta_sigma_theta_e, sigma);
}

model {
    // Estimate the expected mean, sd
    vector[N] mu = latent_mean(Xdc, Xgc, alpha, beta_d, beta_g, beta_c, eid, cid);
    vector[N] sigma_o = latent_sd(Xjc, beta_j, sigma_e, eid);

    // Semi-constructed latent mean of FEC data
    vector[N] mu_f = Xfc * beta_f;

    // Global hyper-prior
    target += half_normal_lpdf(sigma | sigma_sigma);

    // Variable scale priors
    target += std_half_normal_lpdf(eta_sigma_alpha);
    target += std_half_normal_lpdf(eta_sigma_beta_d);
    target += std_half_normal_lpdf(eta_sigma_beta_g);
    target += std_half_normal_lpdf(eta_sigma_beta_j);
    target += std_half_normal_lpdf(eta_sigma_beta_c);
    target += std_half_normal_lpdf(eta_sigma_sigma_e);
    target += std_half_normal_lpdf(eta_sigma_theta_f);
    target += std_half_normal_lpdf(eta_sigma_alpha_f);
    target += std_half_normal_lpdf(eta_sigma_beta_f);
    target += std_half_normal_lpdf(eta_sigma_sigma_f);
    target += std_half_normal_lpdf(eta_sigma_theta_e);

    // Variable offsets
    target += std_normal_lpdf(eta_alpha);
    target += std_normal_lpdf(to_vector(eta_beta_d));
    target += std_normal_lpdf(eta_beta_g);
    target += std_normal_lpdf(eta_beta_j);
    target += std_normal_lpdf(eta_beta_c);
    target += std_normal_lpdf(eta_sigma_e);
    target += std_normal_lpdf(eta_theta_f);
    target += std_normal_lpdf(eta_alpha_f);
    target += std_normal_lpdf(eta_beta_f);
    target += std_normal_lpdf(eta_sigma_f);
    target += std_normal_lpdf(to_vector(eta_theta_e));

    // Likelihood
    if (!prior_check) {
        target += normal_lpdf(Y_logit | mu, sigma_o);
        target += binomial_logit_lpmf(Sf | Kf, theta_f, fec);
        target += binomial_logit_lpmf(Ye | Ke, theta_e);
        target += hurdle_normal_lpdf(Yf | alpha_f, mu_f, sigma_f, fec, eid);
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
        Xffc,
        Xfe,
        Yff,
        alpha,
        alpha_f,
        beta_d,
        beta_g,
        beta_j,
        beta_c,
        beta_f,
        theta_f,
        theta_e,
        eta_sigma_beta_c * sigma,
        sigma_e,
        sigma_f,
        efid,
        cfid,
        fec,
        exid_f,
        exid_d,
        fid,
        cases,
        xe_mean,
        xe_sd
    );

    // Posterior predictive probability of winning
    array[2] vector[M] P_win = win_probability(Y_rep);
    array[2] vector[M] P_win_cf = win_probability(Y_rep_cf);

    // Estimate WAR
    array[2] vector[M] WAR = calculate_WAR(Y_rep, Y_rep_cf);
}