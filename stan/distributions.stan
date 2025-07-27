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

