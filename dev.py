import arviz as az
import polars as pl
import numpy as np
import plotnine as gg
import matplotlib.pyplot as plt

from war.data import WARData
from war.model import WARModel
from war.utils.model import CmdStanModel, clean_dir

war_data = WARData('house').prep_data()
war_fit = (
    WARModel(
        war_data=war_data,
        stan_file='stan/war.stan',
        dir='exe'
    )
    .prep_stan_data()
    .sample(
        iter_warmup=50,
        iter_sampling=50,
        chains=8,
        parallel_chains=8,
        inits=0.01,
        step_size=0.002,
        refresh=20,
        seed=2026
    )
)

model_data = war_data.prepped_data
cids = war_data.cids

cols = model_data.columns
exclusions = [
    'cycle', 'state_name', 'district', 'pct', 'candidate_DEM', 'candidate_REP',
    'cid_DEM', 'cid_REP', 'eid'
]

fixed_effects = [x for x in cols if x not in exclusions]

iid = [
    fixed_effects.index('is_incumbent_DEM') + 1,
    fixed_effects.index('is_incumbent_REP') + 1
]

# Dataset dimensions
N = model_data.shape[0]
E = model_data.unique('eid').shape[0]
C = cids.unique('cid').shape[0]
F = len(fixed_effects)

stan_data = {
    'N': N,
    'E': E,
    'C': C,
    'F': F,
    'X': model_data.select(fixed_effects).to_numpy(),
    'Y': model_data['pct'].to_numpy(),
    'cid': model_data['cid_DEM', 'cid_REP'].to_numpy(),
    'eid': model_data['eid'].to_numpy(),
    'iid': iid,
    'alpha0_mu': 0,
    'alpha0_sigma': 0.15,
    'sigma_alpha_sigma': 0.1,
    'beta_v0_mu': 0,
    'beta_v0_sigma': 0.5,
    'sigma_v_sigma': 0.1,
    'sigma_c_sigma': 0.15,
    'sigma_e_sigma': 0.2,
    'prior_check': 0
}

house_model = CmdStanModel(
    stan_file='stan/war.stan',
    dir='exe'
)

house_fit = house_model.sample(
    data=stan_data,
    iter_warmup=50,
    iter_sampling=50,
    chains=8,
    parallel_chains=8,
    inits=0.01,
    step_size=0.002,
    refresh=20,
    seed=2026
)

candidates = (
    cids
    .unique('cid')
    .with_columns(
        pl.when(pl.col.cid == 1)
        .then(pl.lit('Generic Challenger'))
        .otherwise(pl.col.candidate)
        .alias('candidate')
    )
    .sort('cid')
    ['candidate']
    .to_list()
)

years = (
    model_data
    .unique('eid')
    .sort('cycle')
    ['cycle']
    .to_list()
)

coords = {
    'N': range(N),
    'year': years,
    'candidate': candidates,
    'variable': fixed_effects,
    'Em': years[1:],
    'party': ['dem', 'rep']
}

dims = {
    # Parameters
    'alpha0': [],
    'eta_alpha': ['Em'],
    'sigma_alpha': [],
    'beta_v0': ['variable'],
    'eta_v': ['variable', 'Em'],
    'sigma_v': ['variable'],
    'eta_c': ['candidate'],
    'sigma_c': [],
    'sigma_e': ['year'],

    # Transformed parameters
    'beta_c': ['candidate'],
    'alpha': ['year'],
    'beta_v': ['variable', 'year'],
    'mu': ['N'],
    'sigma': ['N'],

    # Generated quantities
    'Y_rep': ['N'],
    'Y_rep_cf': ['party', 'N'],
    'P_win': ['party', 'N'],
    'P_win_cf': ['party', 'N'],
    'WAR': ['party', 'N']
}

house_az = az.from_cmdstanpy(
    posterior=house_fit,
    coords=coords,
    dims=dims
)



(
    pl.from_pandas(az.summary(house_az, ['alpha', 'beta_v']), include_index=True)
    .rename({'None': 'variable'})
    .with_columns(
        pl.col.variable.str.replace_all('beta_v\\[|\\]|alpha\\[', '')
    )
    .with_columns(
        pl.col.variable.str.split(by=', ')
    )
    .with_columns(
        pl.when(pl.col.variable.map_elements(lambda x: len(x)) == 1)
        .then(pl.lit('alpha'))
        .otherwise(pl.col.variable.map_elements(lambda x: fixed_effects[int(x[0])]))
        .alias('parameter'),
        pl.col.variable.map_elements(lambda x: x[-1]).cast(pl.Int64).alias('eid')
    ) >>
    gg.ggplot(gg.aes(
            x='eid',
            y='mean',
            ymin='hdi_3%',
            ymax='hdi_97%'
    )) + 
    gg.geom_ribbon(alpha=0.25) +
    gg.geom_line() +
    gg.facet_wrap(facets='parameter')
).show()

# # ~aoc as example candidate
(
    house_az
    .posterior
    .WAR
    .sel(WAR_dim_0=0, WAR_dim_1=3097)
    .quantile(q=[0.025, 0.5, 0.975])
)

WAR = (
    house_fit.draws_pd(vars='WAR')
    .pipe(pl.from_pandas, include_index=True)
    .unpivot(
        pl.selectors.all(),
        variable_name='variable',
        value_name='WAR'
    )
    .group_by('variable')
    .agg(pl.implode('WAR'))
    .with_columns(
        pl.col.variable.str.replace_all('WAR\\[|\\]', '').str.split(',')
    )
    .with_columns(
        pl.col.variable.map_elements(lambda x: x[0]).alias('party'),
        pl.col.variable.map_elements(lambda x: x[1]).alias('rowid')
    )
    .select(pl.selectors.exclude('variable'))
    .with_columns(
        pl.when(pl.col.party == '1')
        .then(pl.lit('dem'))
        .otherwise(pl.lit('rep'))
        .alias('party')
    )
    .pivot(
        on='party',
        values='WAR',
        index='rowid'
    )
    .with_columns(pl.col.rowid.cast(pl.Int64))
    .sort('rowid')
    .hstack(house)
    .with_columns(
        pl.col.dem.map_elements(lambda x: x.median()).alias('WAR_med'),
        pl.col.dem.map_elements(lambda x: x.quantile(0.05)).alias('WAR_low'),
        pl.col.dem.map_elements(lambda x: x.quantile(0.95)).alias('WAR_high'),
        pl.concat_str(['state_name', 'district'], separator=' ').alias('group')
    )
)

(
    WAR
    .filter(pl.col.cycle == 2024, pl.col.cid_DEM != 1)
    .select(pl.col(['state_name', 'district', 'candidate_DEM']), pl.selectors.starts_with('WAR'))
    .sort(pl.col.WAR_med, descending=True)
    .filter(pl.col.candidate_DEM.str.contains('Ocasio'))
)

(
    WAR
    .filter(pl.col.candidate_DEM.str.contains('Cuellar')) >>
    gg.ggplot(gg.aes(
        x='cycle',
        y='WAR_med',
        ymin='WAR_low',
        ymax='WAR_high',
        # group='group'
    )) +
    gg.geom_ribbon(alpha=0.25) +
    gg.geom_line()
).show()

(
    WAR >>
    gg.ggplot(gg.aes(
        x='cycle',
        y='WAR_med'
    )) +
    gg.geom_point(alpha=0.25)
).show()

(
    WAR
    .group_by(pl.col.cycle)
    .agg(pl.col.WAR_med.mean()) >>
    gg.ggplot(gg.aes(
        x='cycle',
        y='WAR_med'
    )) +
    gg.geom_line()
).show()

P_win = (
    house_az
    .posterior
    ['P_win']
    .mean(dim=['chain', 'draw'])
)

P_win_cf = (
    house_az
    .posterior
    ['P_win_cf']
    .mean(dim=['chain', 'draw'])
)

(
    pl.DataFrame({
        'WARP': (P_win.to_numpy() - P_win_cf.to_numpy())[1, :]
    })
    .hstack(house)
    .filter(pl.col.cycle == 2024)
    .filter(pl.col.cid_DEM != 1)
    .sort('WARP')
    .select(['state_name', 'district', 'candidate_DEM', 'WARP'])
)

p_win = az.summary(house_az, var_names='P_win')
p_win_cf = az.summary(house_az, var_names='P_win_cf')

(
    p_win
    .pipe(pl.from_pandas, include_index=True)
    .rename({'None': 'variable'})
    .select(pl.col.variable, pl.col.mean.alias('p_win'))
    .hstack(
        p_win_cf.pipe(pl.from_pandas).select(pl.col.mean.alias('p_win_cf')),
    )
    .with_columns(
        pl.col.p_win.sub(pl.col.p_win_cf).alias('WARP')
    )
    .filter(
        pl.col.variable.str.contains('\\[0,')
    )
    .hstack(house)
    .filter(pl.col.cycle == 2024, pl.col.cid_DEM != 1)
    .select(['state_name', 'district', 'candidate_DEM', 'WARP'])
    .sort('WARP', descending=True)
    # .filter(pl.col.candidate_DEM.str.contains('Omar'))
    # >>
    # gg.ggplot(gg.aes(x='WARP')) +
    # gg.geom_histogram(bins=40) +
    # gg.facet_wrap(facets='cycle')
)

(
    p_win_cf
    .pipe(pl.from_pandas, include_index=True)
    .rename({'None': 'variable'})
    .filter(pl.col.variable.str.contains('P_win_cf\\[0'))
    .filter(pl.col.variable.str.contains('3128]'))
)

(
    WAR
    .filter(pl.col.cycle == 2024)
    .filter(pl.col.cid_DEM != 1)
    .sort('WAR_med', descending=True)
    .select(['group', 'candidate_DEM', 'WAR_med'])
    .filter(pl.col.candidate_DEM.str.contains('Omar'))
)

# mondaire jones
tmp_cf = (
    house_fit.draws_pd('Y_rep_cf')
    .pipe(pl.from_pandas, include_index=True)
    .select(['Y_rep_cf[1,3129]'])
)

tmp = (
    house_fit.draws_pd('Y_rep')
    .pipe(pl.from_pandas, include_index=True)
    .select('Y_rep[3129]')
)

(
    tmp
    .hstack(tmp_cf)
    .unpivot(pl.selectors.all()) >>
    gg.ggplot(gg.aes(x='value', color='variable')) +
    gg.geom_density()
).show()

(
    tmp
    .hstack(tmp_cf)
    .with_columns((pl.col('Y_rep[3129]') - pl.col('Y_rep_cf[1,3129]')).alias('WAR')) >>
    gg.ggplot(gg.aes(x='WAR')) +
    gg.geom_density()
).show()

(
    house_az
    .posterior
    ['P_win']
    .mean(dim=['chain', 'draw'])
    .sel(P_win_dim_0=0, P_win_dim_1=3128)
)

(
    P_win
    .sel(P_win_dim_0=0, P_win_dim_1=3128)
)

(
    WAR
    .filter(pl.col.cid_DEM != 1) >>
    gg.ggplot(gg.aes(x='WAR_med')) +
    gg.geom_histogram() +
    gg.facet_wrap(facets='cycle')
).show()

pl.Series.quantile()

Y_rep = (
    az.summary(house_az, 'Y_rep')
    .pipe(pl.from_pandas, include_index=True)
)


az.plot_pair(
    house_az,
    var_names=['beta_v'],
    coords={
        'beta_v_dim_0': [11, 8, 9, 10],
        'beta_v_dim_1': [0]
    },
    show=True
)

az.summary(house_az, var_names='sigma_v')

(
    Y_rep
    .rename({'None': 'rowid'})
    .with_columns(pl.col.rowid.str.replace_all('Y_rep\\[|\\]', '').cast(pl.Int64))
    .join(house.with_row_index('rowid'), on='rowid', how='left')
    .group_by(['state_name', 'district'])
    .agg(pl.implode(['cycle', 'mean', 'hdi_3%', 'hdi_97%']))
    .sample(n=9)
    # .filter(pl.col.state_name == 'California',
    #         pl.col.district == 20)
    .explode(['cycle', 'mean', 'hdi_3%', 'hdi_97%'])
    .with_columns(pl.concat_str(pl.col.state_name, pl.col.district, separator=' ').alias('facet')) >>
    gg.ggplot(gg.aes(
        x='cycle',
        y='mean',
        ymin='hdi_3%',
        ymax='hdi_97%'
    )) +
    gg.geom_ribbon(alpha=0.25) +
    gg.geom_line() +
    gg.facet_wrap(facets='facet')
).show()

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
    # .filter(pl.col.cid != 1)
    .join(candidate_obs, on=['cid', 'candidate'], how='inner')
    .filter(
        pl.col.candidate.str.contains(
            'Golden|Cuellar|Kaptur|Peltola|Perez|Gray|Don Davis|Wild|Caraveo|Mondaire'
        )
    ) >>
    # .sample(n=30) >>
    gg.ggplot(gg.aes(
        x='reorder(candidate, skill_med)',
        y='skill_med',
        ymin='skill_low',
        ymax='skill_high'
    )) +
    gg.geom_pointrange() +
    gg.coord_flip()
).show()

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
        x='pct',
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

(
    az.summary(house_az, var_names='sigma_e')
    .pipe(pl.from_pandas, include_index=True)
    .rename({'None': 'eid'})
    .with_columns(pl.col.eid.str.replace_all('sigma_e\\[|\\]', '').cast(pl.Int64)) >>
    gg.ggplot(gg.aes(
        x='eid',
        y='mean',
        ymin='hdi_3%',
        ymax='hdi_97%'
    )) +
    gg.geom_ribbon(alpha=0.25) +
    gg.geom_line() +
    gg.geom_point()
).show()

(
    pred
    .with_columns(pl.concat_str(['state_name', 'district'], separator=' ').alias('group'))
    .group_by('group')
    .agg(pl.implode(['cycle', 'margin', 'Y_rep_med', 'Y_rep_low', 'Y_rep_high']))
    .sample(n=12)
    .explode(['cycle', 'margin', 'Y_rep_med', 'Y_rep_low', 'Y_rep_high']) >>
    gg.ggplot(gg.aes(
        x='cycle',
        y='Y_rep_med',
        ymin='Y_rep_low',
        ymax='Y_rep_high',
        group='group'
    )) +
    gg.geom_ribbon(alpha=0.25) + 
    gg.geom_line() +
    gg.geom_point(gg.aes(y='margin'), color='royalblue') + 
    gg.facet_wrap(facets='group')
).show()

(
    house
    .filter(pl.col.fec == 0)
    .with_columns(((pl.col.dem_share_fec - pl.col.dem_share_fec.mean()) / pl.col.dem_share_fec.std()).alias('dem_share_fec')) >>
    gg.ggplot(gg.aes(x='dem_share_fec')) + 
    gg.geom_histogram(bins=60)
).show()

az.summary(house_az, 'sigma_c')

clean_dir()