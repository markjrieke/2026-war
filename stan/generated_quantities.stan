/*
    Generate new candidate skill values

    @param N: Integer indicating the number of hypothetical candidates to sample
    @param sigma_c: Candidate group-level standard deviation

    @returns: An N length vector of values with mean 0 and standard deviation
        sigma_c
*/
vector new_candidate_rng(int N,
                         real sigma_c) {
    return to_vector(normal_rng(rep_vector(0, N), sigma_c));
}

/*
    Generate observation-scale posterior predictive samples

    @param mu: A vector of logit-scale means
    @param sigma: A vector of logit-scale standard deviations

    @returns: A vector of observation-scale predictive samples
*/
vector posterior_predictive_rng(vector mu,
                                vector sigma) {
    return inv_logit(to_vector(normal_rng(mu, sigma)));
}

/*
    Generate observation-scale posterior predictive samples

    @param X: A centered/scaled design matrix
    @param alpha: A vector of intercept parameters for each year
    @param beta_v: A matrix of parameter values for each parameter in each year
    @param beta_c: A vector of candidate skill parameters
    @param sigma_e: A vector of logit-scale observation standard deviations
    @param eid: An array of integers mapping year to each race
    @param cid: A multidimensional array of integers mapping the democratic and
        republican candidates to each race

    @returns: A vector of observation-scale predictive samples
*/
vector posterior_predictive_rng(matrix X,
                                vector alpha,
                                matrix beta_v,
                                vector beta_c,
                                vector sigma_e,
                                array[] int eid,
                                array[,] int cid) {
    int N = rows(X);
    vector[N] mu = latent_mean(X, alpha, beta_v, beta_c, eid, cid);
    vector[N] sigma = latent_sd(sigma_e, eid);
    return posterior_predictive_rng(mu, sigma);
}

/*
    Generate observation-scale posterior predictive samples given a
    counterfactual matrix

    @param X: A centered/scaled counterfactual design matrix
    @param alpha: A vector of intercept parameters for each year
    @param beta_v: A matrix of parameter values for each parameter in each year
    @param beta_c: A vector of candidate skill parameters
    @param sigma_c: Candidate group-level standard deviation
    @param sigma_e: A vector of logit-scale observation standard deviations
    @param eid: An array of integers mapping year to each race
    @param cid: A multidimensional array of integers mapping the democratic and
        republican candidates to each race
    @param dem_cf: An integer indicating whether to estimate the predictive
        sample for a democratic counterfactual matrix (1) or republican
        counterfactual matrix (0)

    @returns: A vector of observation-scale counterfactual predictive samples
*/
vector posterior_predictive_cf_rng(matrix X,
                                   vector alpha,
                                   matrix beta_v,
                                   vector beta_c,
                                   real sigma_c,
                                   vector sigma_e,
                                   array[] int eid,
                                   array[,] int cid,
                                   int dem_cf) {
    int N = rows(X);
    vector[N] beta_cf = new_candidate_rng(N, sigma_c);
    vector[N] mu;
    vector[N] sigma;
    vector[N] gamma;
    vector[N] zeta_c;
    vector[N] beta;
    for (n in 1:N) {
        gamma[n] = alpha[eid[n]];
        beta[n] = dot_product(X[n,:], beta_v[:,eid[n]]);
    }
    if (dem_cf) {
        for (n in 1:N) {
            zeta_c[n] = beta_cf[n] - beta_c[cid[n,2]];
        }
    } else {
        for (n in 1:N) {
            zeta_c[n] = beta_c[cid[n,1]] - beta_cf[n];
        }
    }
    mu = gamma + beta + zeta_c;
    sigma = latent_sd(sigma_e, eid);
    return posterior_predictive_rng(mu, sigma);
}

/*
    Generate observatio-scale posterior predictive samples given a set of
    counterfactual matrices

    @param Xc_dem: A centered/scaled counterfactual design matrix for democratic
        candidates
    @param Xc_rep: A centered/scaled counterfactual design matrix for republican
        candidates
    @param alpha: A vector of intercept parameters for each year
    @param beta_v: A matrix of parameter values for each parameter in each year
    @param beta_c: A vector of candidate skill parameters
    @param sigma_c: Candidate group-level standard deviation
    @param sigma_e: A vector of logit-scale observation standard deviations
    @param eid: An array of integers mapping year to each race
    @param cid: A multidimensional array of integers mapping the democratic and
        republican candidates to each race

    @returns: An array of vectors of observation-scale counterfactual predictive
        samples for each party
*/
array[] vector posterior_predictive_cf_rng(matrix Xc_dem,
                                           matrix Xc_rep,
                                           vector alpha,
                                           matrix beta_v,
                                           vector beta_c,
                                           real sigma_c,
                                           vector sigma_e,
                                           array[] int eid,
                                           array[,] int cid) {
    int N = rows(Xc_dem);
    array[2] vector[N] Y_rep_cf;
    Y_rep_cf[1] = posterior_predictive_cf_rng(Xc_dem, alpha, beta_v, beta_c, sigma_c, sigma_e, eid, cid, 1);
    Y_rep_cf[2] = posterior_predictive_cf_rng(Xc_rep, alpha, beta_v, beta_c, sigma_c, sigma_e, eid, cid, 0);
    return Y_rep_cf;
}

/*
    Determine the posterior predictive probability of winning

    Returns an array of vectors with the result of each race based on the
    posterior predictive distribution of Y_rep. The probability is the average
    of results across all draws.

    @param Y_rep: A vector of observation-scale predictive samples

    @returns P_win: An array of vectors containing the posterior predicted
        result in each race.
*/
array[] vector win_probability(vector Y_rep) {
    int N = size(Y_rep);
    array[2] vector[N] P_win;
    for (n in 1:N) {
        P_win[1,n] = Y_rep[n] >= 0.50;
    }
    P_win[2] = 1 - P_win[1];
    return P_win;
}

/*
    Determine the counterfactual posterior predictive probability of winning

    Returns an array of vectors with the result of each race based on the
    posterior predictive distribution of Y_rep_cf. The probability is the average
    of results across all draws.

    @param Y_rep_cf: An array of vectors of observation-scale predictive samples

    @returns P_win_cf: An array of vectors containing the counterfactual
        posterior predicted result in each race.
*/
array[] vector win_probability(array[] vector Y_rep_cf) {
    int N = size(Y_rep_cf[1,:]);
    array[2] vector[N] P_win_cf;
    for (n in 1:N) {
        P_win_cf[1,n] = Y_rep_cf[1,n] >= 0.50;
        P_win_cf[2,n] = Y_rep_cf[2,n] < 0.50;
    }
    return P_win_cf;
}

/*
    Estimate candidate Wins Above Replacement (WAR)

    Calculates the WAR value for each candidate in each race as the difference
    in potential outcomes between the posterior predicted outcome and the
    counterfactual posterior predicted outcome.

    @param Y_rep: A vector of observation-scale predictive samples
    @param Y_rep_cf: An array of vectors of observation-scale predictive samples
*/
array[] vector calculate_WAR(vector Y_rep,
                             array[] vector Y_rep_cf) {
    int N = size(Y_rep);
    array[2] vector[N] WAR;
    WAR[1] = Y_rep - Y_rep_cf[1];
    WAR[2] = Y_rep_cf[2] - Y_rep;
    return WAR;
}
