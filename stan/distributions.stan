/*
    Return the log probability of a half normal with mean 0

    @param sigma: A value (>0)
    @param sigma_sigma: Half-normal standard deviation
*/
real half_normal_lpdf(real sigma,
                      real sigma_sigma) {
    return normal_lpdf(sigma | 0, sigma_sigma) - normal_lccdf(0 | 0, sigma_sigma);
}

/*
    Return the log probability of a half normal with mean 0

    @param sigma: A vector of values (>0)
    @param sigma_sigma: Half-normal standard deviation
*/
real half_normal_lpdf(vector sigma,
                      real sigma_sigma) {
    int D = size(sigma);
    return normal_lpdf(sigma | 0, sigma_sigma) - D * normal_lccdf(0 | 0, sigma_sigma);
}

/*
    Return the log probability of a half normal with mean 0 and sd 1

    @param sigma: A value (>0)
*/
real std_half_normal_lpdf(real sigma) {
    return half_normal_lpdf(sigma | 1);
}

/*
    Return the log probability of a half normal with mean 0 and sd 1

    @param sigma: A vector of values (>0)
*/
real std_half_normal_lpdf(vector sigma) {
    return half_normal_lpdf(sigma | 1);
}

/*
    Return the log probability of a binomial, logit parameterization

    @param Sf: An array of intergers containing successes.
    @param Kf: An array of integers containing trials.
    @param theta_f: A vector of logit-scale probabilities.
    @param fec: An array of integers indicating whether or not FEC filing
        information is available in each year.
*/
real binomial_logit_lpmf(array[] int Sf,
                         array[] int Kf,
                         vector theta_f,
                         array[] int fec) {
    int H = num_elements(Sf);
    real lp = 0.0;
    for (i in 1:H) {
        if (fec[i]) {
            lp += binomial_logit_lpmf(Sf[i] | Kf[i], theta_f[i]);
        }
    }
    return lp;
}

/*
    Return the log probability of a binomial, logit parameterization

    @param Ye: A multidimensional array of "successes" corresponding to
        democratic/republican candidates with prior electoral experience.
    @param Ke: A multidimensional array of trials corresponding to the number
        of non-incumbent democratic/republican candidates
    @param theta_e: A matrix of logit-scale probabilities of a non-incumbent
        democrat/republican having prior electoral experience in each year.
*/
real binomial_logit_lpmf(array[,] int Ye,
                         array[,] int Ke,
                         matrix theta_e) {
    int H = num_elements(Ye[1,:]);
    real lp = 0.0;
    for (p in 1:2) {
        lp += binomial_logit_lpmf(Ye[p] | Ke[p], theta_e[p,1:H]);
    }
    return lp;
}

/*
    Return the log probability of a normal distribution conditional on having
    an observation

    @param Yf: A vector of logit-scale democratic shares of FEC contributions
    @param alpha_f: A vector of time-varying intercept values
    @param mu_f: A vector of time-invariant mean estimates
    @param sigma_f: A vector of time-variying standard deviation values
    @param fec: An array of integers indicating whether or not FEC filing
        information is available in each year.
    @param eid: An array of integers mapping year to observation
*/
real hurdle_normal_lpdf(vector Yf,
                        vector alpha_f,
                        vector mu_f,
                        vector sigma_f,
                        array[] int fec,
                        array[] int eid) {
    int N = size(Yf);
    real lp = 0.0;
    for (n in 1:N) {
        if (fec[eid[n]]) {
            lp += normal_lpdf(Yf[n] | alpha_f[eid[n]] + mu_f[n], sigma_f[eid[n]]);
        }
    }
    return lp;
}
