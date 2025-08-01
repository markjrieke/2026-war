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

    @param Xd: A centered/scaled design matrix for district-level variables
    @param Xg: A centered/scaled design matrix for national-level variables
    @param Xj: A centered/scaled design matrix for sd-affecting variables
    @param alpha: A vector of intercept parameters for each year
    @param beta_d: A matrix of district parameter values for each parameter in
        each year
    @param beta_g: A vector of national parameter values
    @param beta_j: A vector of sd-affecting parameter values
    @param beta_c: A vector of candidate skill parameters
    @param sigma_e: A vector of logit-scale observation standard deviations
    @param eid: An array of integers mapping year to each race
    @param cid: A multidimensional array of integers mapping the democratic and
        republican candidates to each race

    @returns: A vector of observation-scale predictive samples
*/
vector posterior_predictive_rng(matrix Xd,
                                matrix Xg,
                                matrix Xj,
                                vector alpha,
                                matrix beta_d,
                                vector beta_g,
                                vector beta_j,
                                vector beta_c,
                                vector sigma_e,
                                array[] int eid,
                                array[,] int cid) {
    int N = rows(Xd);
    vector[N] mu = latent_mean(Xd, Xg, alpha, beta_d, beta_g, beta_c, eid, cid);
    vector[N] sigma = latent_sd(Xj, beta_j, sigma_e, eid);
    return posterior_predictive_rng(mu, sigma);
}

vector counterfactual_fec_rng(matrix Xf,
                              vector Yf,
                              vector cases,
                              vector theta_f,
                              vector alpha_f,
                              vector beta_f,
                              vector sigma_f,
                              array[] int eid,
                              array[] int fec) {
    int N = rows(Xf);
    real Yf_min = min(Yf);
    real Yf_max = max(Yf);
    array[N] int fec_exist;
    vector[N] fec_share;
    vector[N] fec_result;
    vector[N] theta;
    vector[N] mu = Xf * beta_f;
    vector[N] sigma;
    vector[N] result;

    // Estimate FEC parameters over all obervations
    for (n in 1:N) {
        theta[n] = theta_f[eid[n]];
        mu[n] += alpha_f[eid[n]];
        sigma[n] = sigma_f[eid[n]];
    }

    // Sample hurdle normal
    fec_exist = bernoulli_logit_rng(theta);
    fec_share = to_vector(normal_rng(mu, sigma));
    fec_result = to_vector(fec_exist) .* fec_share;

    // Adjust the share of FEC contributions based on results and each case
    for (n in 1:N) {
        if (cases[n] == 4) {
            result[n] = fec_exist[n] ? fec_result[n] : Yf_min;
        } else if (cases[n] == 3) {
            result[n] = fec_exist[n] ? Yf_max : 0.0;
        } else if (cases[n] == 2) {
            result[n] = fec_exist[n] ? fec_result[n] : Yf_min;
        } else {
            result[n] = fec_exist[n] ? Yf_max : 0.0;
        }
    }

    // Set results to 0 if sampled for years without FEC data
    for (n in 1:N) {
        result[n] *= fec[eid[n]];
    }

    return result;
}

/*
    TODO Add counterfactual estiamtion for replacement candidate experience
vector counterfactual_experience_rng()
*/

/*
    Generate observation-scale posterior predictive samples given a
    counterfactual matrix

    @param Xd: A centered/scaled counterfactual district design matrix
    @param Xg: A centered/scaled counterfacutla national design matrix
    @param Xj: A centered/scaled counterfactual sd-affecting design matrix
    @param alpha: A vector of intercept parameters for each year
    @param beta_d: A matrix of parameter values for each district parameter in
        each year
    @param beta_g: A vector of national parameter values
    @param beta_j: A vector of sd-affecting parameter values
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
vector posterior_predictive_cf_rng(matrix Xd,
                                   matrix Xg,
                                   matrix Xj,
                                   matrix Xf,
                                   vector Yf,
                                   vector alpha,
                                   vector alpha_f,
                                   matrix beta_d,
                                   vector beta_g,
                                   vector beta_j,
                                   vector beta_c,
                                   vector beta_f,
                                   vector theta_f,
                                   real sigma_c,
                                   vector sigma_e,
                                   vector sigma_f,
                                   array[] int eid,
                                   array[,] int cid,
                                   array[] int fec,
                                   vector cases,
                                   int fid,
                                   int dem_cf) {
    int N = rows(Xd);
    int G = cols(Xd);
    matrix[N,G] Xdc = Xd;
    vector[N] beta_cf = new_candidate_rng(N, sigma_c);
    vector[N] mu;
    vector[N] sigma;
    vector[N] gamma;
    vector[N] zeta_c;
    vector[N] beta;

    // Generate counterfacutal estimates for democratic share of FEC contributions
    Xdc[:,fid] = counterfactual_fec_rng(
        Xf, Yf, cases, theta_f, alpha_f, beta_f, sigma_f, eid, fec
    );

    // Estimate parameters for constructing the latent mean
    for (n in 1:N) {
        gamma[n] = alpha[eid[n]];
        beta[n] = dot_product(Xd[n,:], beta_d[:,eid[n]]);
    }

    // Estimate candidate impact
    if (dem_cf) {
        for (n in 1:N) {
            zeta_c[n] = beta_cf[n] - beta_c[cid[n,2]];
        }
    } else {
        for (n in 1:N) {
            zeta_c[n] = beta_c[cid[n,1]] - beta_cf[n];
        }
    }

    // Return posterior predictive based on latent mean, sd
    mu = gamma + beta + zeta_c + Xg * beta_g;
    sigma = latent_sd(Xj, beta_j, sigma_e, eid);
    return posterior_predictive_rng(mu, sigma);
}

/*
    Generate observatio-scale posterior predictive samples given a set of
    counterfactual matrices

    @param Xc_dem: A centered/scaled counterfactual design matrix for democratic
        candidates
    @param Xc_rep: A centered/scaled counterfactual design matrix for republican
        candidates
    @param Xg: A centered/scaled counterfacutla national design matrix
    @param Xj: A centered/scaled counterfactual sd-affecting design matrix
    @param alpha: A vector of intercept parameters for each year
    @param beta_d: A matrix of parameter values for each district parameter in
        each year
    @param beta_g: A vector of national parameter values
    @param beta_j: A vector of sd-affecting parameter values
    @param beta_c: A vector of candidate skill parameters
    @param sigma_c: Candidate group-level standard deviation
    @param sigma_e: A vector of logit-scale observation standard deviations
    @param eid: An array of integers mapping year to each race
    @param cid: A multidimensional array of integers mapping the democratic and
        republican candidates to each race

    @returns: An array of vectors of observation-scale counterfactual predictive
        samples for each party
*/
array[] vector posterior_predictive_cf_rng(matrix Xdc_dem,
                                           matrix Xdc_rep,
                                           matrix Xg,
                                           matrix Xj,
                                           matrix Xf,
                                           vector Yf,
                                           vector alpha,
                                           vector alpha_f,
                                           matrix beta_d,
                                           vector beta_g,
                                           vector beta_j,
                                           vector beta_c,
                                           vector beta_f,
                                           vector theta_f,
                                           real sigma_c,
                                           vector sigma_e,
                                           vector sigma_f,
                                           array[] int eid,
                                           array[,] int cid,
                                           array[] int fec,
                                           int fid,
                                           vector cases) {
    int N = rows(Xdc_dem);
    int D = cols(Xdc_dem);
    array[2] vector[N] Y_rep_cf;
    for (i in 1:2) {
        matrix[N,D] Xdc;
        int dem_flag = 2 - i;
        if (i == 1) {
            Xdc = Xdc_dem;
        } else {
            Xdc = Xdc_rep;
        }
        Y_rep_cf[i] = posterior_predictive_cf_rng(
            Xdc,
            Xg,
            Xj,
            Xf,
            Yf,
            alpha,
            alpha_f,
            beta_d,
            beta_g,
            beta_j,
            beta_c,
            beta_f,
            theta_f,
            sigma_c,
            sigma_e,
            sigma_f,
            eid,
            cid,
            fec,
            cases,
            fid,
            dem_flag
        );

    }
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
