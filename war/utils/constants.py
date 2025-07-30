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
    'is_incumbent_DEM'
]

TIME_INVARIANT_VARIABLES = [
    'intercept',
    'dem_president',
    'midterm',
    'dem_president_midterm',
    'dem_inc_dem_midterm',
    'rep_inc_rep_midterm'
]

SD_VARIABLES = [
    'redistricted',
    'jungle_primary'
]