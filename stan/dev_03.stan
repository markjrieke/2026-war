functions {
    #include functions.stan
}
data {
    // Data dimensions
    int N;                          // Number of observations
    int C;                          // Number of candidates
    int F;                          // Number of fixed-effect variables

    // Observations
    matrix[N,F] X;                  // Fixed-effect variables
    vector[N] Y;                    // Incumbent two-way margin

    // Mapping columns
    array[N,2] int cid;             // Map candidates to races

    // Priors
    real alpha_mu;                  // Intercept prior mean
    real<lower=0> alpha_sigma;      // Intercept prior scale
    real beta_mu;                   // Fixed effect prior mean
    real beta_sigma;                // Fixed effect prior scale
    real<lower=0> sigma_sigma;      // Observation scale prior scale
    real<lower=0> sigma_c_sigma;    // Candidate effect scale prior scale
}

parameters {
    real alpha;                     // Average incumbent margin
    vector[F] beta_v;               // Fixed effects
    vector[C] eta_c;                // Candidate skill parameters
    real<lower=0> sigma_c;          // Candidate skill scale
    real<lower=0> sigma;            // Observation scale
}

transformed parameters {
    // Evaluate hierarchical parameters
    vector[C] beta_c = eta_c * sigma_c;

    // Estimate the expected margin
    vector[N] mu = rep_vector(alpha, N) + X * beta_v;
    {
        vector[N] zeta_c;
        for (n in 1:N) {
            zeta_c[n] = beta_c[cid[n,1]] - beta_c[cid[n,2]];
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
    target += normal_lpdf(Y | mu, sigma);
}