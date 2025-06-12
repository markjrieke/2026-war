real half_normal_lpdf(real sigma,
                      real sigma_sigma) {
    return normal_lpdf(sigma | 0, sigma_sigma) - normal_lccdf(0 | 0, sigma_sigma);
}