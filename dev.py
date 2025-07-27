from typing import List, Union

import arviz as az
import polars as pl
import numpy as np
import plotnine as gg
import matplotlib.pyplot as plt
import xarray as xr

from war.data import WARData
from war.model import WARModel
from war.results import WARResults

war_data = WARData('house').prep_data()
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
war_results.write_full_topline()
war_results.write_publication_topline()
war_results.write_parameter_summaries()

(
    az.from_cmdstanpy(war_fit.war_fit)
    .posterior
    .Y_rep
    .mean(dim=['chain', 'draw'])
    .to_dataframe()
    .pipe(pl.from_pandas, include_index=True)
    .rename({'Y_rep_dim_0': 'M'})
    .join(
        pl.read_parquet('out/summary/mappings/full_data.parquet'),
        on='M',
        how='left'
    )
)
# brain dump for exploratory objs
# - parameter plot over cycle
# - war plot over cycle
# - warp plot over cycle
# - bilateral plot of WAR and WARP
# - interesting candidates over time
# - democrats/republicans war/warp compare
# - posterior predictions relative to results
# - candidate skill vs warp vs war
