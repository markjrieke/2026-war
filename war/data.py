from typing import Literal

from polars import col, lit, read_csv, when
from polars.selectors import exclude

class WARData:

    def __init__(
        self,
        chamber: Literal['house', 'senate']
    ):
        if chamber == 'house':
            self.raw_data = read_csv(
                'data/private/house_forecast_data_updated.csv',
                infer_schema_length=1000
            )
        if chamber == 'senate':
            raise NotImplementedError(
                'Senate model not yet implemented!'
            )

        self.chamber = chamber

    def prep_data(self):

        if self.chamber == 'house':
            self._prep_house_data()

        return self

    def _prep_house_data(self):

        house = self.raw_data

        # Map candidates to races
        mappings = (
            read_csv('data/candidate_filters.csv')
            .with_columns(col.exclusions.fill_null('y'))
            .filter(col.exclusions != 'x')
            .select(exclude('exclusions'))
            .with_columns(col.is_incumbent == 'x')
            .with_columns(col.is_incumbent.fill_null(False))
            .select(exclude(['state', 'seat']))
        )

        # Model dataframe with candidates in wide format
        house = (
            house
            .filter(col.uncontested == 0)
            .select([
                'cycle', 'state_name', 'district', 'pct',
                'age', 'income', 'colplus', 'urban', 'asian', 'black', 'hispanic',
                'dem_pres_twop_lag_lean_one', 'experience', 'logit_dem_share_fec'
            ])
            .with_columns(
                when(col.experience < 0).then(lit(1)).otherwise(lit(0)).alias('exp_disadvantage'),
                when(col.experience > 0).then(lit(1)).otherwise(lit(0)).alias('exp_advantage')
            )
            .select(exclude('experience'))
            .join(
                mappings.select(exclude('politician_id')),
                on=['cycle', 'state_name', 'district'],
                how='inner'
            )
            .pivot(on='party', values=['candidate', 'is_incumbent'])
        )

        # Set of candidates who have won a race during the modeled period
        winners = (
            house
            .select(
                col.cycle,
                col.state_name,
                col.district,
                when(col.pct >= 0.50)
                .then(col.candidate_DEM)
                .otherwise(col.candidate_REP)
                .alias('candidate')
            )
            .join(
                mappings.select(['cycle', 'state_name', 'district', 'candidate', 'politician_id']),
                on=['cycle', 'state_name', 'district', 'candidate'],
                how='inner'
            )
        )

        # Set of incumbents during the modeled period
        incumbents = (
            house
            .select(
                col.cycle,
                col.state_name,
                col.district,
                when(col.is_incumbent_DEM)
                .then(col.candidate_DEM)
                .when(col.is_incumbent_REP)
                .then(col.candidate_REP)
                .alias('candidate')
            )
            .filter(col.candidate.is_not_null())
            .join(
                mappings.select(['cycle', 'state_name', 'district', 'candidate', 'politician_id']),
                on=['cycle', 'state_name', 'district', 'candidate'],
                how='inner'
            )
        )

        # Politician IDs of candidates who either were incumbents or won a race
        # during the modeled time period
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
                when(col.politician_id.is_in(named_candidates))
                .then(col.politician_id)
                .otherwise(lit(0))
                .rank('dense')
                .alias('cid')
            )
        )

        # Join in mapping ids
        base_cols = ['cycle', 'state_name', 'district']
        house = (
            house
            .join(
                mappings.select(base_cols + ['candidate', 'politician_id']),
                left_on=base_cols + ['candidate_DEM'],
                right_on=base_cols + ['candidate'],
                how='left'
            )
            .join(
                cids.select(['politician_id', 'cid']),
                on='politician_id',
                how='left'
            )
            .select(exclude('politician_id'))
            .rename({'cid': 'cid_DEM'})
            .join(
                mappings.select(base_cols + ['candidate', 'politician_id']),
                left_on=base_cols + ['candidate_REP'],
                right_on=base_cols + ['candidate'],
                how='left'
            )
            .join(
                cids.select(['politician_id', 'cid']),
                on='politician_id',
                how='left'
            )
            .select(exclude('politician_id'))
            .rename({'cid': 'cid_REP'})
            .with_columns(col.cycle.rank('dense').alias('eid'))
        )

        self.prepped_data = house
        self.cids = cids

