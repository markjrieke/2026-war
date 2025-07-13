import arviz as az
import polars as pl
import numpy as np
import plotnine as gg

from war.utils.model import CmdStanModel, clean_dir

# Base set of seats to model exclude uncontested seats
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

# Map candidates to races
mappings = (
    pl.read_csv('data/candidate_filters.csv')
    .with_columns(pl.col.exclusions.fill_null('y'))
    .filter(pl.col.exclusions != 'x')
    .select(pl.selectors.exclude('exclusions'))
    .with_columns(pl.col.is_incumbent == 'x')
    .with_columns(pl.col.is_incumbent.fill_null(False))
    .select(pl.selectors.exclude(['state', 'seat']))
    .join(
        house.select(['cycle', 'state_name', 'district']),
        on=['cycle', 'state_name', 'district'],
        how='inner'
    )
)

# Get results in terms of the incumbent party's performance with candidates
# in a wide format
house = (
    mappings
    .select(pl.selectors.exclude('politician_id'))
    .join(
        house,
        on=['cycle', 'state_name', 'district'],
        how='inner'
    )
    .pivot(on='party', values=['candidate', 'is_incumbent'])
    .with_columns(
        pl.when((pl.col.incumbent_party == 'new_seat') &
                (pl.col.is_incumbent_DEM))
        .then(pl.lit('DEM'))
        .when((pl.col.incumbent_party == 'new_seat') &
              (pl.col.is_incumbent_REP))
        .then(pl.lit('REP'))
        .otherwise(pl.col.incumbent_party)
        .alias('incumbent_party')
    )
    .with_columns(
        pl.when(pl.col.incumbent_party == 'new_seat')
        .then(pl.lit('DEM'))
        .otherwise(pl.col.incumbent_party)
        .alias('incumbent_party')
    )
    .with_columns(
        pl.when(pl.col.incumbent_party == 'REP')
        .then((1 - pl.col.pct - 0.5) * 2)
        .otherwise((pl.col.pct - 0.5) * 2)
        .alias('inc_margin'),
        pl.when(pl.col.incumbent_party == 'DEM')
        .then(pl.col.is_incumbent_DEM)
        .otherwise(pl.col.is_incumbent_REP)
        .alias('inc_running')
    )
    .select(pl.selectors.exclude(pl.selectors.starts_with('is_')))
    .select(pl.selectors.exclude('pct'))
)

# Set of candidates who have won a race during the modeled period
winners = (
    house
    .select(
        pl.col.cycle,
        pl.col.state_name,
        pl.col.district,
        pl.when((pl.col.incumbent_party == 'DEM') &
                (pl.col.inc_margin > 0))
        .then(pl.col.candidate_DEM)
        .when((pl.col.incumbent_party == 'REP') &
              (pl.col.inc_margin < 0))
        .then(pl.col.candidate_DEM)
        .otherwise(pl.col.candidate_REP)
        .alias('candidate')
    )
    .join(
        mappings.select(['cycle', 'state_name', 'district', 'candidate', 'politician_id']),
        on=['cycle', 'state_name', 'district', 'candidate'],
        how='inner'
    )
    .unique('politician_id')
    .sort(['candidate', 'politician_id'])
    .select(['candidate', 'politician_id'])
)

# Set of incumbents during the modeled period
incumbents = (
    house
    .select(
        pl.col.cycle,
        pl.col.state_name,
        pl.col.district,
        pl.when((pl.col.incumbent_party == 'DEM') &
                (pl.col.inc_running))
        .then(pl.col.candidate_DEM)
        .when((pl.col.incumbent_party == 'REP') &
              (pl.col.inc_running))
        .then(pl.col.candidate_REP)
        .alias('candidate')
    )
    .filter(pl.col.candidate.is_not_null())
    .join(
        mappings.select(['cycle', 'state_name', 'district', 'candidate', 'politician_id']),
        on=['cycle', 'state_name', 'district', 'candidate'],
        how='inner'
    )
    .unique('politician_id')
    .sort(['candidate', 'politician_id'])
    .select(['candidate', 'politician_id'])
)

# Politician IDs of candidates who either were incumbents or won a race during
# the modeled period
named_candidates = (
    winners
    .vstack(incumbents)
    .unique('politician_id')
    .sort(['candidate', 'politician_id'])
    ['politician_id']
)

# Create a Stan-friendly mapping of candidates
# Generic challengers are mapped to position 1
cids = (
    mappings
    .unique('politician_id')
    .sort(['candidate', 'politician_id'])
    .select(['candidate', 'politician_id'])
    .with_columns(
        pl.when(pl.col.politician_id.is_in(named_candidates))
        .then(pl.col.politician_id)
        .otherwise(pl.lit(0))
        .alias('cid')
    )
    .with_columns(
        pl.col.cid.rank('dense')
    )
)

# Join ID mappings
house = (
    house
    .join(
        mappings.select(['cycle', 'state_name', 'district', 'candidate', 'politician_id']),
        left_on=['cycle', 'state_name', 'district', 'candidate_DEM'],
        right_on=['cycle', 'state_name', 'district', 'candidate'],
        how='left'
    )
    .join(
        cids.select(['politician_id', 'cid']),
        on='politician_id',
        how='left'
    )
    .select(pl.selectors.exclude('politician_id'))
    .rename({'cid': 'cid_DEM'})
    .join(
        mappings.select(['cycle', 'state_name', 'district', 'candidate', 'politician_id']),
        left_on=['cycle', 'state_name', 'district', 'candidate_REP'],
        right_on=['cycle', 'state_name', 'district', 'candidate'],
        how='left'
    )
    .join(
        cids.select(['politician_id', 'cid']),
        on='politician_id',
        how='left'
    )
    .select(pl.selectors.exclude('politician_id'))
    .rename({'cid': 'cid_REP'})
)

# Frame candidates & IDs in terms of incumbency
house = (
    house
    .with_columns(
        pl.when(pl.col.incumbent_party == 'DEM')
        .then(pl.col.candidate_DEM)
        .otherwise(pl.col.candidate_REP)
        .alias('inc_party_candidate'),
        pl.when(pl.col.incumbent_party == 'DEM')
        .then(pl.col.candidate_REP)
        .otherwise(pl.col.candidate_DEM)
        .alias('opp_party_candidate'),
        pl.when(pl.col.incumbent_party == 'DEM')
        .then(pl.col.cid_DEM)
        .otherwise(pl.col.cid_REP)
        .alias('inc_cid'),
        pl.when(pl.col.incumbent_party == 'DEM')
        .then(pl.col.cid_REP)
        .otherwise(pl.col.cid_DEM)
        .alias('opp_cid')
    )
    .select(pl.selectors.exclude(pl.selectors.ends_with('_DEM')))
    .select(pl.selectors.exclude(pl.selectors.ends_with('_REP')))
)

stan_data = {
    'N': house.shape[0],
    'C': cids.unique('cid').shape[0],
    'Y': house['inc_margin'].to_numpy(),
    'cid': house['inc_cid', 'opp_cid'].to_numpy(),
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

house_az = az.from_cmdstanpy(posterior=house_fit)

# ~aoc as example candidate
(
    house_az
    .posterior
    .beta_c
    .sel(beta_c_dim_0=809)
    .quantile(q=[0.025, 0.5, 0.975])
)

clean_dir()