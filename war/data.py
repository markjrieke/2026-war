import arviz as az
import polars as pl
import numpy as np

from polars import col, read_csv, when, Float64

from war.utils.model import CmdStanModel, clean_dir

class WARData:

    # TODO Model config options
    def __init__(self):
        self.raw_data = read_csv(
            'data/private/house_forecast_data.csv',
            infer_schema_length=1000
        )

    # TODO Prep based on model config options
    def prep_data(self):
        self.prepped_data = (
            self.raw_data
            .filter(col.uncontested == 0)
            .with_columns(
                when(col.pct == 'NA').then(None).otherwise(col.pct).cast(Float64).alias('dem_pct'),
                when(col.other_pct == 'NA').then(None).otherwise(col.other_pct).cast(Float64).alias('other_pct')
            )
            .filter((col.dem_pct.is_not_null()), (col.other_pct.is_not_null()))
            .with_columns(
                (1 - col.dem_pct - col.other_pct).alias('rep_pct')
            )
            .with_columns(
                (col.dem_pct - col.rep_pct).alias('margin'),
            )
            .select(
                ['margin', 'age', 'income', 'colplus', 'urban', 'asian', 'black',
                'hispanic', 'dem_inc_party', 'dem_share_fec']
            )
        )

        return self

