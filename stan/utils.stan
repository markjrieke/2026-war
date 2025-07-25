matrix standardize(matrix X) {
    int N = rows(X);
    int F = cols(X);
    matrix[N,F] Xc;
    vector[F] X_mean;
    vector[F] X_sd;
    for (f in 1:F) {
        X_mean[f] = mean(X[:,f]);
        X_sd[f] = sd(X[:,f]);
        Xc[:,f] = (X[:,f] - X_mean[f]) / X_sd[f];
    }
    return Xc;
}

matrix standardize_cf(matrix X,
                      int iid) {
    int N = rows(X);
    int F = cols(X);
    matrix[N,F] Xc;
    vector[F] X_mean;
    vector[F] X_sd;

    // Create centered design matrix
    for (f in 1:F) {
        X_mean[f] = mean(X[:,f]);
        X_sd[f] = sd(X[:,f]);
        Xc[:,f] = (X[:,f] - X_mean[f]) / X_sd[f];
    }

    // Replace columns in iid with centered zeroes
    for (f in 1:F) {
        if (f == iid) {
            Xc[:,f] = rep_vector(-X_mean[f] / X_sd[f], N);
        }
    }

    return Xc;
}

vector random_walk(real theta0,
                   vector eta,
                   real sigma) {
    int E = size(eta) + 1;
    int Em = E - 1;
    vector[E] theta;
    theta[1] = theta0;
    theta[2:E] = eta * sigma;
    for (em in 1:Em) {
        theta[em+1] += theta[em];
    }
    return theta;
}

matrix random_walk(vector theta0,
                   matrix eta,
                   vector sigma) {
    int F = size(theta0);
    int E = cols(eta) + 1;
    int Em = E - 1;
    matrix[F,E] theta;
    theta[:,1] = theta0;
    theta[:,2:E] = diag_pre_multiply(sigma, eta);
    for (em in 1:Em) {
        theta[:,em+1] += theta[:,em];
    }
    return theta;
}

vector latent_mean(matrix X,
                   vector alpha,
                   matrix beta_v,
                   vector beta_c,
                   array[] int eid,
                   array[,] int cid) {
    int N = rows(X);
    vector[N] mu;
    vector[N] gamma;
    vector[N] zeta_c;
    vector[N] beta;
    for (n in 1:N) {
        gamma[n] = alpha[eid[n]];
        zeta_c[n] = beta_c[cid[n,1]] - beta_c[cid[n,2]];
        beta[n] = dot_product(X[n,:], beta_v[:,eid[n]]);
    }
    mu = gamma + beta + zeta_c;
    return mu;
}

vector latent_sd(vector sigma_e,
                 array[] int eid) {
    int N = size(eid);
    vector[N] sigma;
    for (n in 1:N) {
        sigma[n] = sigma_e[eid[n]];
    }
    return sigma;
}