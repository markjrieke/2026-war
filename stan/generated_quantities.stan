vector new_candidate_rng(int N,
                         real sigma_c) {
    return to_vector(normal_rng(rep_vector(0, N), sigma_c));
}

vector posterior_predictive_rng(vector mu,
                                vector sigma) {
    return inv_logit(to_vector(normal_rng(mu, sigma)));
}

vector posterior_predictive_cf_rng(matrix X,
                                   vector alpha,
                                   matrix beta_v,
                                   vector beta_c,
                                   real sigma_c,
                                   vector sigma,
                                   array[] int eid,
                                   array[,] int cid,
                                   int dem_cf) {
    int N = rows(X);
    vector[N] beta_cf = new_candidate_rng(N, sigma_c);
    vector[N] mu;
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
    return posterior_predictive_rng(mu, sigma);
}

array[] vector posterior_predictive_cf_rng(matrix Xc_dem,
                                           matrix Xc_rep,
                                           vector alpha,
                                           matrix beta_v,
                                           vector beta_c,
                                           real sigma_c,
                                           vector sigma,
                                           array[] int eid,
                                           array[,] int cid) {
    int N = rows(Xc_dem);
    array[2] vector[N] Y_rep_cf;
    Y_rep_cf[1] = posterior_predictive_cf_rng(Xc_dem, alpha, beta_v, beta_c, sigma_c, sigma, eid, cid, 1);
    Y_rep_cf[2] = posterior_predictive_cf_rng(Xc_rep, alpha, beta_v, beta_c, sigma_c, sigma, eid, cid, 0);
    return Y_rep_cf;
}

array[] vector win_probability(vector Y_rep) {
    int N = size(Y_rep);
    array[2] vector[N] P_win;
    for (n in 1:N) {
        P_win[1,n] = Y_rep[n] >= 0.50;
    }
    P_win[2] = 1 - P_win[1];
    return P_win;
}

array[] vector win_probability(array[] vector Y_rep_cf) {
    int N = size(Y_rep_cf[1,:]);
    array[2] vector[N] P_win_cf;
    for (n in 1:N) {
        P_win_cf[1,n] = Y_rep_cf[1,n] >= 0.50;
        P_win_cf[2,n] = Y_rep_cf[2,n] < 0.50;
    }
    return P_win_cf;
}

array[] vector calculate_WAR(vector Y_rep,
                             array[] vector Y_rep_cf) {
    int N = size(Y_rep);
    array[2] vector[N] WAR;
    WAR[1] = Y_rep - Y_rep_cf[1];
    WAR[2] = Y_rep_cf[2] - Y_rep;
    return WAR;
}
