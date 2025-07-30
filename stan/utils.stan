/*
    Center and scale the columns of a matrix, X

    @param X: A design matrix

    @returns Xc: A centered/scaled matrix of the same dimensions as X.
*/
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

/*
    Center and scale the columns of a matrix, X, based on the mean/sd of columns
    in a matrix Z

    @param X: A counterfactual design matrix
    @param Z: A design matrix

    @returns Xc: A centered/scaled matrix of the same dimensions as X.
*/
matrix standardize(matrix X,
                   matrix Z) {
    int M = rows(X);
    int F = cols(X);
    matrix[M,F] Xc;
    vector[F] Z_mean;
    vector[F] Z_sd;
    for (f in 1:F) {
        Z_mean[f] = mean(Z[:,f]);
        Z_sd[f] = sd(Z[:,f]);
        Xc[:,f] = (X[:,f] - Z_mean[f]) / Z_sd[f];
    }
    return Xc;
}

/*
    Create a centered/scaled counterfactual matrix.

    Centers and scales a matrix, X, based on the means and standard deviations
    of each column in matrix Z, then sets all values in the column that flags
    incumbency, `iid`, to the centered/scaled value for `0`. I.e., this
    counterfactual matrix is the same as `standardize(X, Z)` with no incumbents
    in `iid`.

    @param X: A counterfactual design matrix
    @param Z: A design matrix
    @param iid: An integer indicating the column in the design matrix that
        corresponds to incumbency.

    @returns Xc: A centered/scaled matrix of the same dimensions as X with the
        values in column `iid` taking on the centered/scaled value for `0`.
*/
matrix standardize_cf(matrix X,
                      matrix Z,
                      int iid) {
    int M = rows(X);
    int F = cols(X);
    matrix[M,F] Xc;
    vector[F] Z_mean;
    vector[F] Z_sd;

    // Create centered design matrix
    for (f in 1:F) {
        Z_mean[f] = mean(Z[:,f]);
        Z_sd[f] = sd(Z[:,f]);
        Xc[:,f] = (X[:,f] - Z_mean[f]) / Z_sd[f];
    }

    // Replace columns in iid with centered zeroes
    for (f in 1:F) {
        if (f == iid) {
            Xc[:,f] = rep_vector(-Z_mean[f] / Z_sd[f], M);
        }
    }

    return Xc;
}

/*
    Add an intercept term to the first column of a matrix

    @param X: a design matrix

    @returns Xi: A with on addtional column of 1s added to X
*/
matrix add_intercept(matrix X) {
    int N = rows(X);
    int C = cols(X) + 1;
    matrix[N,C] Xi;
    Xi[:,1] = rep_vector(1, N);
    Xi[:,2:C] = X;
    return Xi;
}

/*
    Propagate a random walk over a vector

    @param theta0: The initial state of the random walk
    @param eta: A vector of N-1 standardized random walk steps
    @param sigma: The random walk scale

    @returns theta: A vector of length N that randomly drifts from the initial
        state.
*/
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

/*
    Propagate a random walk over a matrix

    This function assumes that each random walk is uncorrelated. In essence, it
    performs the same function as the vector random walk implementation en-masse.

    @param theta0: A length F vector of initial states for each random walk
    @param eta: A matrix of size F, N-1 standardized random walk steps for each
        parameter.
    @param sigma: A length F vector of random walk scales

    @returns theta: A matrix containing uncorrelated random walks for F variables
        across N steps.
*/
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

/*
    Estimate the logit-scale mean democratic two-party voteshare

    @param X: A centered/scaled design matrix
    @param alpha: A vector of intercept parameters for each year
    @param beta_d: A matrix of parameter values for each parameter in each year
    @param beta_c: A vector of candidate skill parameters
    @param eid: An array of integers mapping year to each race
    @param cid: A multidimensional array of integers mapping the democratic and
        republican candidates to each race

    @returns mu: A vector of logit-scale mean democratic two-party voteshare
        estimates
*/
vector latent_mean(matrix Xd,
                   matrix Xg,
                   vector alpha,
                   matrix beta_d,
                   vector beta_g,
                   vector beta_c,
                   array[] int eid,
                   array[,] int cid) {
    int N = rows(Xd);
    vector[N] mu;
    vector[N] gamma;
    vector[N] zeta_c;
    vector[N] beta;
    for (n in 1:N) {
        gamma[n] = alpha[eid[n]];
        zeta_c[n] = beta_c[cid[n,1]] - beta_c[cid[n,2]];
        beta[n] = dot_product(Xd[n,:], beta_d[:,eid[n]]);
    }
    mu = gamma + beta + zeta_c + Xg * beta_g;
    return mu;
}

/*
    Estimate the logit-scale standard deviation round the estimate for the
    democratic two-party voteshare

    @param Xjc #TODO
    @param beta_j #TODO
    @param sigma_e: A vector of logit-scale observation standard deviations for
        each year
    @param eid: An array of integers mapping year to each race

    @returns sigma: A vector of logit-scale standard deviation estimates
*/
vector latent_sd(matrix Xjc,
                 vector beta_j,
                 vector sigma_e,
                 array[] int eid) {
    int N = size(eid);
    vector[N] sigma = Xjc * beta_j;
    for (n in 1:N) {
        sigma[n] += sigma_e[eid[n]];
    }
    return exp(sigma);
}