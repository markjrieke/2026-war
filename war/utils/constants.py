DEFAULT_PRIORS = {
    'sigma_sigma': 1
}

TIME_VARYING_VARIABLES = [
    'age',
    'income',
    'colplus',
    'urban',
    'asian',
    'black',
    'hispanic',
    'dem_pres_twop_lag_lean_one',
    'logit_dem_share_fec',
    'exp_disadvantage',
    'exp_advantage',
    'is_incumbent_REP',
    'is_incumbent_DEM',
    'presidential_party',
    'midterm',
    'president_midterm'
]

TIME_INVARIANT_VARIABLES = [
    'intercept',
    'inc_party_same_party_pres_midterm'
]

SD_VARIABLES = [
    'redistricted',
]

FEC_VARIABLES = [
    'exp_disadvantage',
    'exp_advantage',
    'is_incumbent_REP',
    'is_incumbent_DEM'
]