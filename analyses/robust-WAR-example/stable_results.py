from os.path import exists

import polars as pl

from war.data import WARData
from war.model import WARModel
from war.results import WARResults

from war.utils.constants import TIME_VARYING_VARIABLES
from war.utils.transformations import from_xarray

# These variables are required to simulate the counterfactual candidates
exclusions = [
    'dem_share_fec',
    'exp_advantage',
    'exp_disadvantage',
    'is_incumbent_DEM',
    'is_incumbent_REP'
]

# We'll loop over these variables, dropping each one once
variables = [x for x in TIME_VARYING_VARIABLES if x not in exclusions]

# Use the same dataset each time
war_data = WARData('house').prep_data()

# Utility function for calculating/saving RMSE
def write_rmse(
    war_results: WARResults,
    variable: str
) -> None:

    # Get predictions for calculating RMSE
    predictions = (
        war_results.idata.posterior['Y_rep'].quantile(q=0.5, dim=['chain', 'draw'])
    )
    predictions = from_xarray(predictions)

    # Calculate RMSE
    fit_summary = (
        predictions
        .pivot(on='quantile', values='Y_rep')
        .rename({'0.5': 'Y_rep'})
        .join(
            war_results.war_fit.full_data,
            on='M',
            how='left'
        )
        .filter(pl.col.uncontested == 0)
        .group_by(pl.lit(variable).alias('excluded_variable'))
        .agg(
            (((pl.col.Y_rep - pl.col.pct).pow(2)).sum() / pl.len()).pow(0.5).alias('rmse')
        )
    )

    # Write results out
    filepath = 'analyses/robust-WAR-example/fit_summary.parquet'
    if not exists(filepath):
        fit_summary.write_parquet(filepath)
    else:
        (
            fit_summary
            .vstack(pl.read_parquet(filepath))
            .write_parquet(filepath)
        )

for variable in variables:
    print('=====================================')
    print(f'Refitting model without {variable}')
    print('=====================================')

    # Exclude `variable` from being used in the model
    model_variables = TIME_VARYING_VARIABLES.copy()
    model_variables.remove(variable)

    # Fit the model without `variable`
    war_fit = (
        WARModel(
            war_data=war_data,
            stan_file='stan/war.stan',
            dir='exe'
        )
        .prep_stan_data(time_varying_vars=model_variables)
        .sample(
            iter_warmup=100,
            iter_sampling=100,
            chains=10,
            parallel_chains=10,
            inits=0.01,
            step_size=0.002,
            refresh=20,
            seed=2026
        )
    )

    # Write fit results from each model
    war_results = WARResults(war_fit)
    war_results._write_parameter_summary(
        posterior=war_results.idata.posterior,
        file=f'analyses/robust-WAR-example/variables/Y_rep-excluding-{variable}.parquet',
        variable='Y_rep'
    )

    # Write RMSE results
    write_rmse(war_results=war_results, variable=variable)

# Fit one last time to all variables to use 'None' as a comparison point
war_fit = (
    WARModel(
        war_data=war_data,
        stan_file='stan/war.stan',
        dir='exe'
    )
    .prep_stan_data()
    .sample(
        iter_warmup=100,
        iter_sampling=100,
        chains=10,
        parallel_chains=10,
        inits=0.01,
        step_size=0.002,
        refresh=20,
        seed=2026
    )
)

war_results = WARResults(war_fit)
write_rmse(war_results=war_results, variable='None')