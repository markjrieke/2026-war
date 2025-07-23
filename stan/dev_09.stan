functions {
    #include functions.stan
}
data {
    // Data dimensions
    int N;                          // Number of observations
    int E;                          // Number of election cycles
    int C;                          // Number of candidates
    int F;                          // Number of non-candidate variables

    // Observations
    matrix[N,F] X;                  // Design matrix
    vector[N] Y;                    // Incumbent two-way margin

    // Mapping columns
    array[N,2] int cid;             // Map candidates to races
    array[N] int eid;               // Map election cyle to race

    // Priors
    real alpha_mu;                  // Intercept prior mean
    real<lower=0> alpha_sigma;      // Intercept prior scale
    real beta_v0_mu;                // Non-candidate parameter initial state prior mean
    real beta_v0_sigma;             // Non-candidate parameter initial state prior scale
    real<lower=0> sigma_v_sigma;    // Non-candidate parameter random walk scale
    real<lower=0> sigma_c_sigma;    // Candidate effect scale prior scale
    real<lower=0> sigma_e_sigma;    // Observation prior scale

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

    // Number of random walk steps
    int Em = E - 1;
}

parameters {
    real alpha;                     // Average incumbent margin
    vector[F] beta_v0;              // Non-candidate parameter random walk initial state
    matrix[F,Em] eta_v;             // Non-candidate parameter random walk
    vector<lower=0>[F] sigma_v;     // Non-candidate parameter random walk scale
    vector[C] eta_c;                // Candidate skill parameters
    real<lower=0> sigma_c;          // Candidate skill scale
    vector<lower=0>[E] sigma_e;     // Observation scale
}

transformed parameters {
    // Evaluate hierarchical parameters
    vector[C] beta_c = eta_c * sigma_c;

    // Evaluate the random walk
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
        vector[N] zeta_c;
        vector[N] beta;
        for (n in 1:N) {
            zeta_c[n] = beta_c[cid[n,1]] - beta_c[cid[n,2]];
            beta[n] = dot_product(Xc[n,:], beta_v[:,eid[n]]);
            sigma[n] = sigma_e[eid[n]];
        }
        mu = rep_vector(alpha, N) + beta + zeta_c;
    }
}

model {
    // Priors
    target += normal_lpdf(alpha | alpha_mu, alpha_sigma);
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
    array[N] real Y_rep = normal_rng(mu, sigma);
}