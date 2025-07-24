functions {
    #include functions.stan
}
data {
    // Data dimensions
    int N;                              // Number of observations
    int E;                              // Number of election cycles
    int C;                              // Number of candidates
    int F;                              // Number of non-candidate variables

    // Observations
    matrix[N,F] X;                      // Design matrix
    vector[N] Y;                        // Incumbent two-way margin

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
    matrix[N,F] Xc;
    vector[F] X_mean;
    vector[F] X_sd;
    for (f in 1:F) {
        X_mean[f] = mean(X[:,f]);
        X_sd[f] = sd(X[:,f]);
        Xc[:,f] = (X[:,f] - X_mean[f]) / X_sd[f];
    }

    // Create contrast matrices that eschew incumbency
    matrix[N,F] Xc_dem = Xc;
    matrix[N,F] Xc_rep = Xc;
    for (f in 1:F) {
        if (f == iid[1]) {
            Xc_dem[:,f] = rep_vector(-X_mean[f] / X_sd[f], N);
        }
        if (f == iid[2]) {
            Xc_rep[:,f] = rep_vector(-X_mean[f] / X_sd[f], N);
        }
    }

    // Number of random walk steps
    int Em = E - 1;
}

parameters {
    real alpha0;                        // Average incumbent margin initial state
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

    // Evaluate the random walk for the intercept
    vector[E] alpha;
    alpha[1] = alpha0;
    alpha[2:E] = eta_alpha * sigma_alpha;
    for (em in 1:Em) {
        alpha[em+1] += alpha[em];
    }

    // Evaluate the random walk over parameters
    matrix[F,E] beta_v;
    beta_v[:,1] = beta_v0;
    beta_v[:,2:E] = diag_pre_multiply(sigma_v, eta_v);
    for (em in 1:Em) {
        beta_v[:,em+1] += beta_v[:,em];
    }

    // Estimate the expected mean, sd
    vector[N] mu;
    vector[N] sigma;
    {
        vector[N] gamma;
        vector[N] zeta_c;
        vector[N] beta;
        for (n in 1:N) {
            gamma[n] = alpha[eid[n]];
            zeta_c[n] = beta_c[cid[n,1]] - beta_c[cid[n,2]];
            beta[n] = dot_product(Xc[n,:], beta_v[:,eid[n]]);
            sigma[n] = sigma_e[eid[n]];
        }
        mu = gamma + beta + zeta_c;
    }
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
        target += normal_lpdf(Y | mu, sigma);
    }
}

generated quantities {
    // Posterior retrodictive
    vector[N] Y_rep = to_vector(normal_rng(mu, sigma));
    array[2] vector[N] P_win;
    for (n in 1:N) {
        P_win[1,n] = Y_rep[n] > 0;
    }
    P_win[2] = 1 - P_win[1];

    vector[N] beta_c_dem = to_vector(to_vector(normal_rng(rep_vector(0, N), 1)) * sigma_c);
    vector[N] beta_c_rep = to_vector(to_vector(normal_rng(rep_vector(0, N), 1)) * sigma_c);
    vector[N] mu_dem;
    vector[N] mu_rep;
    {
        vector[N] gamma;
        vector[N] zeta_c_dem;
        vector[N] zeta_c_rep;
        vector[N] beta_dem;
        vector[N] beta_rep;
        for (n in 1:N) {
            gamma[n] = alpha[eid[n]];
            zeta_c_dem[n] = beta_c_dem[n] - beta_c[cid[n,2]];
            zeta_c_rep[n] = beta_c[cid[n,1]] - beta_c_rep[n];
            beta_dem[n] = dot_product(Xc_dem[n,:], beta_v[:,eid[n]]);
            beta_rep[n] = dot_product(Xc_rep[n,:], beta_v[:,eid[n]]);
        }
        mu_dem = gamma + beta_dem + zeta_c_dem;
        mu_rep = gamma + beta_rep + zeta_c_rep;
    }

    array[2] vector[N] Y_rep_cf;
    Y_rep_cf[1] = to_vector(normal_rng(mu_dem, sigma));
    Y_rep_cf[2] = to_vector(normal_rng(mu_rep, sigma));

    array[2] vector[N] WAR;
    for (p in 1:2) {
        WAR[p] = Y_rep - Y_rep_cf[p];
    }

    array[2] vector[N] P_win_cf;
    for (n in 1:N) {
        P_win_cf[1,n] = Y_rep_cf[1,n] > 0;
        P_win_cf[2,n] = Y_rep_cf[2,n] < 0;
    }
}