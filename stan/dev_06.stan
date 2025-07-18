functions {
    #include functions.stan
}
data {
    // Data dimensions
    int N;                          // Number of observations
    int E;                          // Number of election cycles
    int C;                          // Number of candidates
    int F;                          // Number of fixed-effect variables

    // Observations
    matrix[N,F] X;                // Design matrix (mean)
    vector[N] Y;                    // Incumbent two-way margin

    // Mapping columns
    array[N,2] int cid;             // Map candidates to races
    array[N] int eid;               // Map election cyle to race

    // Priors
    real alpha_mu;                  // Intercept prior mean
    real<lower=0> alpha_sigma;      // Intercept prior scale
    real beta_mu;                   // Fixed effect prior mean
    real beta_sigma;                // Fixed effect prior scale
    real<lower=0> sigma_sigma;      // Observation scale prior scale
    real<lower=0> sigma_c_sigma;    // Candidate effect scale prior scale

    int<lower=0, upper=1> prior_check;
}

transformed data {
    // Center the design matrix
    matrix[N,F] Xc;
    vector[F] X_mean;
    for (f in 1:F) {
        X_mean[f] = mean(X[:,f]);
        Xc[:,f] = X[:,f] - X_mean[f];
    }
}

parameters {
    real alpha;                     // Average incumbent margin
    vector[F] beta_v;               // Fixed effects on mean
    vector[C] eta_c;                // Candidate skill parameters
    real<lower=0> sigma_c;          // Candidate skill scale
    vector<lower=0>[E] sigma_e;     // Observation scale
}

transformed parameters {
    // Evaluate hierarchical parameters
    vector[C] beta_c = eta_c * sigma_c;

    // Estimate the expected mean, sd
    vector[N] mu = rep_vector(alpha, N) + Xc * beta_v;
    vector[N] sigma;
    {
        vector[N] zeta_c;
        for (n in 1:N) {
            zeta_c[n] = beta_c[cid[n,1]] - beta_c[cid[n,2]];
            sigma[n] = sigma_e[eid[n]];
        }
        mu += zeta_c;
    }
}

model {
    // Priors
    target += normal_lpdf(alpha | alpha_mu, alpha_sigma);
    target += normal_lpdf(beta_v | beta_mu, beta_sigma);
    target += half_normal_lpdf(sigma | sigma_sigma);
    target += std_normal_lpdf(eta_c);
    target += half_normal_lpdf(sigma_c | sigma_c_sigma);

    // Likelihood
    if (!prior_check) {
        target += normal_lpdf(Y | mu, sigma);
    }
}

generated quantities {
    array[N] real Y_rep = normal_rng(mu, sigma);
}