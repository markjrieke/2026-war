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
        'cycle', 'state_name', 'district', 'incumbent_party', 'pct',
        'age', 'income', 'colplus', 'urban', 'asian', 'black', 'hispanic',
        'dem_pres_twop_lag_lean_one', 'dem_pres_twop_lag_lean_two'
    ])
    .join(
        pl.read_csv('data/presidential_party.csv'),
        on='cycle',
        how='left'
    )
    .with_columns(
        (pl.col.incumbent_party == pl.col.presidential_party)
        .alias('inc_party_same_party_pres')
    )
    .select(pl.selectors.exclude('presidential_party'))
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
        pl.col.pct.sub(0.5).mul(2).alias('margin')
    )
    .select(pl.selectors.exclude('pct'))
)

# Set of candidates who have won a race during the modeled period
winners = (
    house
    .select(
        pl.col.cycle,
        pl.col.state_name,
        pl.col.district,
        pl.when(pl.col.margin >= 0)
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
        pl.when(pl.col.is_incumbent_DEM)
        .then(pl.col.candidate_DEM)
        .when(pl.col.is_incumbent_REP)
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
    .with_columns(
        pl.col.cycle.rank('dense').alias('eid')
    )
)

fixed_effects = [
    'age', 'income', 'colplus', 'urban', 'asian', 'black', 'hispanic',
    'is_incumbent_DEM', 'is_incumbent_REP', 'inc_party_same_party_pres',
    'dem_pres_twop_lag_lean_one', 'dem_pres_twop_lag_lean_two'
]

stan_data = {
    'N': house.shape[0],
    'E': house.unique('eid').shape[0],
    'C': cids.unique('cid').shape[0],
    'F': len(fixed_effects),
    'X': house.select(fixed_effects).to_numpy(),
    'Y': house['margin'].to_numpy(),
    'cid': house['cid_DEM', 'cid_REP'].to_numpy(),
    'eid': house['eid'].to_numpy(),
    'alpha_mu': 0,
    'alpha_sigma': 0.1,
    'beta_v0_mu': 0,
    'beta_v0_sigma': 0.25,
    'sigma_v_sigma': 0.1,
    'sigma_c_sigma': 0.025,
    'sigma_e_sigma': 0.1,
    'prior_check': 0
}

house_model = CmdStanModel(
    stan_file='stan/dev_08.stan',
    dir='exe'
)

house_fit = house_model.sample(
    data=stan_data,
    iter_warmup=1000,
    iter_sampling=1000,
    chains=8,
    parallel_chains=8
)

print(house_fit.diagnose())

house_az = az.from_cmdstanpy(posterior=house_fit)

(
    pl.from_pandas(az.summary(house_az, 'beta_v'), include_index=True)
    .rename({'None': 'variable'})
    .with_columns(
        pl.col.variable.str.replace_all('beta_v\\[|\\]', '')
    )
    .with_columns(
        pl.col.variable.str.split(by=', ')
    )
    .with_columns(
        pl.col.variable.map_elements(lambda x: x[0]).cast(pl.Int64).alias('parameter'),
        pl.col.variable.map_elements(lambda x: x[1]).cast(pl.Int64).alias('eid')
    ) >>
    gg.ggplot(gg.aes(
            x='eid',
            y='mean',
            ymin='hdi_3%',
            ymax='hdi_97%'
    )) + 
    gg.geom_ribbon(alpha=0.25) +
    gg.facet_wrap(facets='parameter')
).show()

# # ~aoc as example candidate
(
    house_az
    .posterior
    .beta_c
    .sel(beta_c_dim_0=809)
    .quantile(q=[0.025, 0.5, 0.975])
)

(
    pl.from_pandas(house_fit.draws_pd('Y_rep'))
    .with_row_index('.draw')
    .unpivot(
        pl.selectors.starts_with('Y_rep'),
        index='.draw',
        variable_name='rowid',
        value_name='Y_rep'
    )
    .group_by('rowid')
    .agg(pl.implode('Y_rep'))
    .with_columns(
        pl.col.rowid.str.replace_all('Y_rep\\[|\\]', '').cast(pl.Int64)
    )
    .slice(offset=0, length=1)
    .explode('Y_rep') >>
    gg.ggplot(gg.aes(x='Y_rep')) +
    gg.geom_density()
).show()

candidate_obs = (
    house
    .select(pl.selectors.starts_with('cid'))
    .unpivot(on=pl.selectors.all(), value_name='cid')
    .group_by('cid')
    .agg(pl.len().alias('n'))
    .join(cids, on='cid', how='left')
    .select(['cid', 'candidate', 'n'])
    .sort('cid')
    .filter(pl.col.cid != 1)
    .filter(pl.col.n > 1)
)

(
    pl.from_pandas(house_fit.draws_pd('beta_c'))
    .with_row_index('.draw')
    .unpivot(
        pl.selectors.starts_with('beta_c'),
        index='.draw',
        variable_name='cid',
        value_name='beta_c'
    )
    .group_by('cid')
    .agg(pl.implode('beta_c'))
    .with_columns(
        pl.col.cid.str.replace_all('beta_c\\[|\\]', '').cast(pl.Int64)
    )
    .join(cids, on='cid', how='left')
    .with_columns(
        pl.col.beta_c.map_elements(lambda x: x.median()).alias('skill_med'),
        pl.col.beta_c.map_elements(lambda x: x.quantile(0.05)).alias('skill_low'),
        pl.col.beta_c.map_elements(lambda x: x.quantile(0.95)).alias('skill_high')
    )
    .filter(pl.col.cid == 1)
    # .join(candidate_obs, on=['cid', 'candidate'], how='inner')
    # .sample(n=30) >>
    # gg.ggplot(gg.aes(
    #     x='reorder(candidate, skill_med)',
    #     y='skill_med',
    #     ymin='skill_low',
    #     ymax='skill_high'
    # )) +
    # gg.geom_pointrange() +
    # gg.coord_flip()
)

pred = (
    pl.from_pandas(house_fit.draws_pd('Y_rep'))
    .with_row_index('.draw')
    .unpivot(
        pl.selectors.starts_with('Y_rep'),
        index='.draw',
        variable_name='rowid',
        value_name='Y_rep'
    )
    .group_by('rowid')
    .agg(pl.implode('Y_rep'))
    .with_columns(
        pl.col.rowid.str.replace_all('Y_rep\\[|\\]', '').cast(pl.Int64)
    )
    .sort('rowid')
    .hstack(house)
    .with_columns(
        pl.col.Y_rep.map_elements(lambda x: x.median()).alias('Y_rep_med'),
        pl.col.Y_rep.map_elements(lambda x: x.quantile(0.05)).alias('Y_rep_low'),
        pl.col.Y_rep.map_elements(lambda x: x.quantile(0.95)).alias('Y_rep_high')
    )
)

(
    pred >>
    gg.ggplot(gg.aes(
        x='margin',
        y='Y_rep_med',
        ymin='Y_rep_low',
        ymax='Y_rep_high'
    )) + 
    gg.geom_pointrange(alpha=0.05) +
    gg.facet_wrap(facets='cycle') + 
    gg.geom_abline(linetype='dashed', color='red')
).show()

(
    pred
    .with_columns(
        ((pl.col.margin > pl.col.Y_rep_low) &
         (pl.col.margin < pl.col.Y_rep_high)).alias('in')
    )
    .group_by(['cycle', 'in'])
    .agg(pl.len().alias('n'))
    .with_columns(
        (pl.col.n / pl.col.n.sum().over('cycle')).alias('pct')
    )
    .filter(pl.col('in'))
    .sort('cycle') >>
    gg.ggplot(gg.aes(x='cycle', y='pct')) + 
    gg.geom_point()
).show()


clean_dir()