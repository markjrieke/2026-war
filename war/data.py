from typing import Literal

from polars import DataFrame, col, lit, read_csv, when
from polars.selectors import all, exclude, starts_with

from war.utils.constants import STATES

class WARData:

    def __init__(
        self,
        chamber: Literal['house', 'senate']
    ):

        """
        Utility class for importing and preparing datasets for modeling.

        The original raw data is curated by [G. Elliott Morris](https://www.gelliottmorris.com/).
        It contains two-party results for each congressional election from
        2000-2024, excluding special elections, along with a host of potentially
        relevant predictor variables (district demographics, partisanship,
        FEC fundraising, etc.).

        The dataset is [slightly modified](https://github.com/markjrieke/2026-war/issues/2)
        for use in the class. Namely, jungle primary results are overwritten
        with general results when there is no majority winner in the primary,
        vacancies, new districts, and purely independent incumbents have the
        incumbent party reclassed, and minor inconsistencies with results are
        resolved.

        While the rest of the project is open source, these datasets have been
        asked to remain private.

        Parameters
        ----------
        chamber : Literal['house', 'senate']
            The chamber of congress to prep data for.
        """

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

        """
        Prepare a dataset for passing to Stan for modeling.

        This method attaches several DataFrames to the WARData object:
        * **full_data** : A prepped dataset including results for all elections.
        * **prepped_data** : A prepped dataset excluding uncontested elections.
        * **cids** : A DataFrame mapping each candidate to a `cid` (candidate ID).

        Many candidates appear in the dataset once to challenge an incumbent, but
        do not win and do not run again. As such, only "named candidates" are
        given a `cid`. Named candidates include candidates who either win an
        election or appear as incumbents during the modeled time period. The
        latter qualification accounts for incumbent candidates who lose in 2000
        as well as candidates who win in off-year special elections to fill
        vacancies but then lose in the regularly scheduled congressional race.
        """

        if self.chamber == 'house':
            self._prep_house_data()

        return self

    def _prep_house_data(self):

        """Internal method that performs prep for house datasets"""

        house = self.raw_data

        # State DataFrame for converting abbreviations to names
        states = (
            DataFrame(STATES)
            .unpivot(
                all(),
                variable_name='state',
                value_name='state_name'
            )
        )

        # Create holistic set of candidate experience mappings
        candidate_experience = (
            read_csv('data/private/house_candidates_updated.csv')
            .select(exclude('effective_party'))
            .vstack(read_csv('data/private/house_candidates_historical_updated.csv'))
            .rename({'seat': 'district'})
            .join(states, on='state', how='left')
            .select(['cycle', 'state_name', 'district', 'politician_id', 'experience'])
        )

        # Map candidates to races
        mappings = (
            read_csv('data/candidate_filters.csv')
            .with_columns(col.exclusions.fill_null('y'))
            .filter(col.exclusions != 'x')
            .select(exclude('exclusions'))
            .with_columns(col.is_incumbent == 'x')
            .with_columns(col.is_incumbent.fill_null(False))
            .select(exclude(['state', 'seat']))
            .join(
                candidate_experience,
                on=['cycle', 'state_name', 'district', 'politician_id'],
                how='left'
            )
        )

        # Model dataframe with candidates in wide format
        house = (
            house
            .select([
                'cycle', 'state_name', 'district', 'pct', 'uncontested',
                'age', 'income', 'colplus', 'urban', 'asian', 'black', 'hispanic',
                'dem_pres_twop_lag_lean_one', 'dem_share_fec',
                'redistricted', 'incumbent_party', 'has_fec'
            ])
            .join(
                mappings.select(exclude('politician_id')),
                on=['cycle', 'state_name', 'district'],
                how='inner'
            )
            .pivot(on='party', values=['candidate', 'is_incumbent', 'experience'])
            .with_columns(
                col('experience_DEM', 'experience_REP').fill_null(0)
            )
            .with_columns(
                (col.experience_DEM > col.experience_REP).alias('exp_advantage'),
                (col.experience_DEM < col.experience_REP).alias('exp_disadvantage')
            )
            .with_columns(
                starts_with('candidate').fill_null('Uncontested'),
                starts_with('is_incumbent').fill_null(False)
            )
            .join(
                read_csv('data/jungle_primaries.csv'),
                on=['cycle', 'state_name', 'district'],
                how='left'
            )
            .with_columns(
                (col.n_democratic_candidates.is_not_null() |
                 col.n_republican_candidates.is_not_null())
                 .alias('jungle_primary')
            )
            .with_columns(col.dem_share_fec.add(0.5))
            .with_columns(
                when((~col.jungle_primary) & (col.has_fec == 1) & (col.uncontested == 0))
                .then(col.dem_share_fec)
                .otherwise(lit(0.5))
                .alias('dem_share_fec')
            )
            .select(exclude(starts_with('n_')))
            .join(
                read_csv('data/presidential_party.csv'),
                on='cycle',
                how='left'
            )
            .with_columns(
                when((col.presidential_party == 'DEM') &
                     (col.incumbent_party == 'DEM') &
                     (col.midterm))
                .then(lit(1))
                .when((col.presidential_party == 'REP') &
                      (col.incumbent_party == 'REP') &
                      (col.midterm))
                .then(lit(-1))
                .otherwise(lit(0))
                .alias('inc_party_same_party_pres_midterm')
            )
            .select(exclude('incumbent_party'))
            .with_columns(
                when(col.presidential_party == 'DEM')
                .then(lit(1))
                .otherwise(lit(-1))
                .alias('presidential_party'),
                when(col.midterm)
                .then(lit(1))
                .otherwise(lit(-1))
                .alias('midterm'),
                when((col.presidential_party == 'DEM') & col.midterm)
                .then(lit(1))
                .when((col.presidential_party == 'REP') & col.midterm)
                .then(lit(-1))
                .otherwise(lit(0))
                .alias('president_midterm')
            )
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

        # Set of politicians who appear in the dataset multiple times
        repeat_candidates = (
            house
            .select(['cycle', 'state_name', 'district', 'candidate_DEM', 'candidate_REP'])
            .unpivot(
                on=['candidate_DEM', 'candidate_REP'],
                variable_name='party',
                value_name='candidate',
                index=['cycle', 'state_name', 'district']
            )
            .select(exclude('party'))
            .join(
                mappings.select(['cycle', 'state_name', 'district', 'candidate', 'politician_id']),
                on=['cycle', 'state_name', 'district', 'candidate'],
                how='inner'
            )
            .filter(col.politician_id.count().over('candidate') > 1)
        )

        # Politician IDs of candidates who either were incumbents or won a race
        # during the modeled time period
        named_candidates = (
            winners
            .vstack(incumbents)
            .vstack(repeat_candidates)
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
            .select(['candidate', 'politician_id', 'party'])
            .with_columns(
                when(col.politician_id.is_in(named_candidates))
                .then(col.politician_id)
                .when(col.party == 'DEM')
                .then(lit(0))
                .otherwise(lit(-1))
                .rank('dense')
                .alias('cid')
            )
            .select(exclude('party'))
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
            .with_columns(col.cid.fill_null(lit(2)))
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
            .with_columns(col.cid.fill_null(lit(1)))
            .select(exclude('politician_id'))
            .rename({'cid': 'cid_REP'})
            .with_columns(col.cycle.rank('dense').alias('eid'))
            .with_columns(
                when(col.cid_DEM == 2).then(lit(1)).otherwise(lit(0)).alias('generic_DEM'),
                when(col.cid_REP == 1).then(lit(1)).otherwise(lit(0)).alias('generic_REP')
            )
        )

        prepped_data = (
            house
            .filter(col.uncontested == 0)
            .select(exclude('uncontested'))
        )

        self.full_data = house
        self.prepped_data = prepped_data
        self.cids = cids

