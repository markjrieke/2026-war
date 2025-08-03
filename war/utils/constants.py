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
    'exp_advantage',
    'exp_disadvantage',
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
    'exp_advantage',
    'exp_disadvantage',
    'is_incumbent_DEM',
    'is_incumbent_REP'
]

STATES = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'DC': 'District of Columbia',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
}