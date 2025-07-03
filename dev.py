import arviz as az
import polars as pl
import numpy as np
import plotnine as gg

from war.utils.model import CmdStanModel, clean_dir

house = (
    pl.read_csv(
        'data/private/house_forecast_data_updated.csv',
        infer_schema_length=1000
    )
    .filter(pl.col.uncontested == 0)
    .select([
        'cycle', 'state_name', 'district', 'incumbent_party', 'pct'
    ])
)

filters = (
    pl.read_csv('data/candidate_filters.csv')
    .with_columns(pl.col.exclusions.fill_null('y'))
    .filter(pl.col.exclusions != 'x')
    .select([
        'cycle', 'state_name', 'district', 'party', 'candidate', 'politician_id'
    ])
)

all_candidates = (
    filters
    .join(
        house.select(['cycle', 'state_name', 'district']),
        on=['cycle', 'state_name', 'district'],
        how='inner'
    )
    .unique('politician_id')
    .sort(['candidate', 'politician_id'])
    .select(['candidate', 'politician_id'])
)

(
    filters
    .join(
        house.select([
            'cycle', 'state_name', 'district',
            'incumbent_party', 'pct'
        ]),
        on=['cycle', 'state_name', 'district'],
        how='inner'
    )
    .select(pl.selectors.exclude('candidate'))
    .pivot(on='party', values='politician_id')
    .join(all_candidates, left_on='DEM', right_on='politician_id')
    .rename({'candidate': 'candidate_dem'})
    .join(all_candidates, left_on='REP', right_on='politician_id')
    .rename({'candidate': 'candidate_rep'})
    .rename({
        'DEM': 'pid_dem',
        'REP': 'pid_rep'
    })
    .with_columns(
        (pl.col.pct >= 0.5).alias('inc_dem'),
        (pl.col.pct < 0.5).alias('inc_rep')
    )
)

candidates = (
    filters
    .join(
        house.select(['cycle', 'state_name', 'district']),
        on=['cycle', 'state_name', 'district'],
        how='inner'
    )
    .select(pl.selectors.exclude('candidate'))
    .pivot(on='party', values='politician_id')
    .join(cid, left_on='DEM', right_on='politician_id')
    .rename({
        'cid': 'cid_dem',
        'candidate': 'candidate_dem'
    })
    .join(cid, left_on='REP', right_on='politician_id')
    .rename({
        'cid': 'cid_rep',
        'candidate': 'candidate_rep'
    })
    .select(pl.selectors.exclude(['DEM', 'REP']))
)

(
    house
    .join(candidates, on=['cycle', 'state_name', 'district'])
)

cid = (
    filters
    .join(
        house.select(['cycle', 'state_name', 'district']),
        on=['cycle', 'state_name', 'district'],
        how='inner'
    )
    .unique('politician_id')
    .sort(['candidate', 'politician_id'])
    .select(['candidate', 'politician_id'])
    .with_row_index('cid')
    .with_columns(cid = pl.col.cid + 1)
)

# blegh ------------------------------------------------------------------------

# deal with new_seat l8r dawg
model_data = (
    house
    .join(candidates, on=['cycle', 'state_name', 'district'], how='left')
    .with_columns(
        inc_margin = 2 * (pl.col.pct - 0.5)
    )
    .with_columns(
        pl.when(pl.col.incumbent_party.is_in(['DEM', 'new_seat']))
        .then(pl.col.inc_margin)
        .otherwise(-pl.col.inc_margin)
        .alias('inc_margin')
    )
    .select(
        pl.col.inc_margin.alias('Y'),
        pl.when(pl.col.incumbent_party.is_in(['DEM', 'new_seat']))
            .then(pl.col.cid_dem)
            .otherwise(pl.col.cid_rep)
            .alias('cid_inc'),
        pl.when(~pl.col.incumbent_party.is_in(['DEM', 'new_seat']))
            .then(pl.col.cid_dem)
            .otherwise(pl.col.cid_rep)
            .alias('cid_opp')
    )
)

stan_data = {
    'N': house.shape[0],
    'C': cid.shape[0],
    'Y': model_data['Y'].to_numpy(),
    'cid': model_data['cid_inc', 'cid_opp'].to_numpy(),
    'alpha_mu': 0.05,
    'alpha_sigma': 0.10,
    'sigma_sigma': 0.10,
    'sigma_c_sigma': 0.10
}

house_model = CmdStanModel(
    stan_file='stan/dev_02.stan',
    dir='exe'
)

house_fit = house_model.sample(
    data=stan_data,
    iter_warmup=1000,
    iter_sampling=1000,
    chains=8,
    parallel_chains=8
)

plot = az.plot_posterior(house_fit, var_names='alpha')
plot.imshow()

2+3

(
    house
    .join(candidates, on=['cycle', 'state_name', 'district'], how='left')
    .filter(pl.col.incumbent_party == 'new_seat')
    .sort(['state_name', 'district', 'cycle'])
)

(
    house
    .group_by('incumbent_party')
    .agg(n = pl.col.incumbent_party.count())
)

(
    house
    .select('race_id')
    .join(candidates, on='race_id', how='left')
    .group_by(['race_id', 'party'])
    .len()
    .filter(pl.col.party.is_in(['DEM', 'REP']))
    .group_by('race_id')
)

(
    house
    .join(candidates, on='race_id', how='left')
    .select(
        ['cycle', 'state_name', 'district', 'state', 'seat', 'party', 'candidate',
         'politician_id']
    )
    .sort(['cycle', 'state_name', 'district', 'party'])
    .write_csv('data/candidate_filters.csv')
)

house_fit = house_model.sample(
    data=stan_data,
    iter_warmup=1000,
    iter_sampling=1000,
    chains=8,
    parallel_chains=8
)

house_az = az.from_cmdstanpy(posterior=house_fit)

clean_dir()