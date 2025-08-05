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
    vector[M] cases;
    for (m in 1:M) {
        if (Yff_inv_logit[m] > 0.99) {
            cases[m] = 1;               // D ~ 1
        } else if (Yff_inv_logit[m] < 0.01) {
            cases[m] = 2;               // R ~ 1
        } else if (Yff_inv_logit[m] == 0.5) {
            cases[m] = 3;               // Even
        } else {
            cases[m] = 4;               // Mix
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
    real<lower=0> eta_sigma_alpha;
    vector<lower=0>[D] eta_sigma_beta_d;
    real<lower=0> eta_sigma_beta_g;
    real<lower=0> eta_sigma_beta_j;
    real<lower=0> eta_sigma_beta_c;
    real<lower=0> eta_sigma_sigma_e;
    real<lower=0> eta_sigma_theta_f;
    real<lower=0> eta_sigma_alpha_f;
    real<lower=0> eta_sigma_beta_f;
    real<lower=0> eta_sigma_sigma_f;
    vector[2] eta_sigma_theta_e;

    // Random walk initial states
    real eta_alpha0;
    vector[D] eta_beta_d0;
    real eta_sigma_e0;
    real eta_theta_f0;
    real eta_alpha_f0;
    real eta_sigma_f0;
    vector[2] eta_theta_e0;

    // Random walk steps
    vector[Em] eta_alpha;
    matrix[D,Em] eta_beta_d;
    vector[G] eta_beta_g;
    vector[J] eta_beta_j;
    vector[C] eta_beta_c;
    vector[Em] eta_sigma_e;
    vector[Em] eta_theta_f;
    vector[Em] eta_alpha_f;
    vector[F] eta_beta_f;
    vector[Em] eta_sigma_f;
    matrix[2,Em] eta_theta_e;
}

transformed parameters {
    // Evaluate hierarchical parameters
    vector[C] beta_c = eta_beta_c * eta_sigma_beta_c * sigma;
    vector[G] beta_g = eta_beta_g * sigma;
    vector[J] beta_j = eta_beta_j * sigma;
    vector[F] beta_f = eta_beta_f * sigma;

    // Conditionally use additional hierarchical parameter if necessary
    if (G > 1) {
        beta_g *= eta_sigma_beta_g;
    }

    if (J > 1) {
        beta_j *= eta_sigma_beta_j;
    }

    if (F > 1) {
        beta_f *= eta_sigma_beta_f;
    }

    // Evaluate random walk over the intercept
    vector[E] alpha;
    alpha[1] = eta_alpha0 * sigma;
    alpha[2:E] = eta_alpha * eta_sigma_alpha * sigma;
    for (i in 2:E) {
        alpha[i] += alpha[i-1];
    }

    // Evaluate random walk over the parameters
    matrix[D,E] beta_d;
    beta_d[:,1] = eta_beta_d0 * sigma;
    beta_d[:,2:E] = diag_pre_multiply(eta_sigma_beta_d * sigma, eta_beta_d);
    for (i in 2:E) {
        beta_d[:,i] += beta_d[:,i-1];
    }

    // Evaluate random walk over the standard deviations
    vector[E] sigma_e;
    sigma_e[1] = eta_sigma_e0 * sigma;
    sigma_e[2:E] = eta_sigma_e * eta_sigma_sigma_e * sigma;
    for (i in 2:E) {
        sigma_e[i] += sigma_e[i-1];
    }

    // Evaluate random walk over FEC filing probability
    vector[E] theta_f;
    theta_f[1] = eta_theta_f0 * sigma;
    theta_f[2:E] = eta_theta_f * eta_sigma_theta_f * sigma;
    for (i in 2:E) {
        theta_f[i] += theta_f[i-1];
    }

    // Evaluate random walk over FEC filing share
    vector[E] alpha_f;
    vector[E] sigma_f;
    alpha_f[1] = eta_alpha_f0 * sigma;
    sigma_f[1] = eta_sigma_f0 * sigma;
    alpha_f[2:E] = eta_alpha_f * eta_sigma_alpha_f * sigma;
    sigma_f[2:E] = eta_sigma_f * eta_sigma_sigma_f * sigma;
    for (i in 2:E) {
        alpha_f[i] += alpha_f[i-1];
        sigma_f[i] += sigma_f[i-1];
    }
    sigma_f = exp(sigma_f);

    // Evaluate random walk over candidate experience
    matrix[2,E] theta_e;
    theta_e[:,1] = eta_theta_e0 * sigma;
    theta_e[:,2:E] = diag_pre_multiply(eta_sigma_theta_e * sigma, eta_theta_e);
    for (i in 2:E) {
        theta_e[:,i] += theta_e[:,i-1];
    }
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
        for (i in 1:H) {
            if (fec[i]) {
                target += binomial_logit_lpmf(Sf[i] | Kf[i], theta_f[i]);
            }
            for (p in 1:2) {
                target += binomial_logit_lpmf(Ye[p] | Ke[p], theta_e[p,1:H]);
            }
        }
        for (n in 1:N) {
            if (fec[eid[n]]) {
                target += normal_lpdf(Yf[n] | alpha_f[eid[n]] + mu_f[n], sigma_f[eid[n]]);
            }
        }
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