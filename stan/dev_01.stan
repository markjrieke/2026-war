functions {
    #include functions.stan
}
data {
    // Data dimensions
    int N;              // Number of observations
    int K;              // Number of covariates

    // Observations
    matrix[N, K] X;     // Design matrix
    vector[N] Y;        // Democratic two-way margin

    // Priors
    real alpha_mu;
    real<lower=0> alpha_sigma;
    vector[K] beta_mu;
    vector<lower=0>[K] beta_sigma;
    real<lower=0> sigma_sigma;
}

parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu = alpha + X * beta;
}

model {
    // Priors
    target += normal_lpdf(alpha | alpha_mu, alpha_sigma);
    target += normal_lpdf(beta | beta_mu, beta_sigma);
    target += half_normal_lpdf(sigma | sigma_sigma);

    // Likelihood
    target += normal_lpdf(Y | mu, sigma);
}
