from typing import List, Union
from os.path import join

from arviz import InferenceData, from_cmdstanpy
from polars import DataFrame, col, implode, lit, when
from polars.selectors import exclude
from xarray import Dataset

from war.model import WARModel
from war.utils.transformations import from_xarray

class WARResults:

    def __init__(
        self,
        war_fit: WARModel
    ):

        self.war_fit = war_fit
        self.idata = self._extract_idata()

    def write_full_topline(
        self,
        path: str = 'out/summary',
        cred_level: float = 0.9
    ):

        self._extract_topline(cred_level).write_parquet(join(path, 'full_topline.parquet'))

    def write_publication_topline(
        self,
        path: str = 'out/summary',
        cred_level: float = 0.9
    ):

        (
            self._extract_topline(cred_level)
            .filter(col.cycle == 2024)
            .select(exclude('cycle'))
            .write_csv(join(path, 'current_topline.csv'))
        )

    def write_parameter_summaries(
        self,
        path: str = 'out/summary/variables',
        cred_levels: Union[List[float], float] = [0.66, 0.8, 0.95]
    ):

        # Write out all variables contained in the fit object
        for variable in self.idata.posterior.data_vars:
            self._write_parameter_summary(
                posterior=self.idata.posterior,
                file=join(path, f'{variable}.parquet'),
                variable=variable,
                cred_levels=cred_levels
            )

        # Calculate and write WARP manually
        WARP = self._extract_WARP()
        WARP.write_parquet(join(path, 'WARP.parquet'))

    def _extract_topline(
        self,
        cred_level: float
    ) -> DataFrame:

        quantiles = self._set_quantiles(cred_level)
        WAR = self.idata.posterior['WAR'].quantile(q=quantiles, dim=['chain', 'draw'])
        WARP = self._extract_WARP()

        WAR = (
            from_xarray(WAR)
            .group_by(['M', 'party'])
            .agg(implode(['quantile', 'WAR']))
            .pivot(on='party', values='WAR')
            .sort('M')
            .rename({
                'dem': 'WAR_DEM',
                'rep': 'WAR_REP'
            })
        )

        topline = (
            WARP
            .pivot(on='party', values='WARP')
            .rename({
                'dem': 'WARP_DEM',
                'rep': 'WARP_REP'
            })
            .hstack(self.war_fit.war_data.full_data)
            .join(WAR, on='M', how='left')
            .select(
                col(['cycle', 'state_name', 'district', 'pct', 'uncontested']),
                when(col.pct >= 0.5).then(lit('DEM')).otherwise(lit('REP')).alias('party'),
                when(col.pct >= 0.5).then(col.candidate_DEM).otherwise(col.candidate_REP).alias('representative'),
                when(col.pct >= 0.5).then(col.WARP_DEM).otherwise(col.WARP_REP).alias('WARP'),
                when(col.pct >= 0.5).then(col.WAR_DEM).otherwise(col.WAR_REP).alias('WAR')
            )
            .with_columns(
                col.WAR.map_elements(lambda x: x[0]).alias('WAR'),
                col.WAR.map_elements(lambda x: x[1]).alias('WAR_lower'),
                col.WAR.map_elements(lambda x: x[2]).alias('WAR_upper')
            )
            .select([
                'cycle', 'state_name', 'district', 'uncontested', 'pct',
                'representative', 'party', 'WARP', 'WAR', 'WAR_lower', 'WAR_upper'
            ])
        )

        return topline

    def _extract_WARP(self) -> DataFrame:
        p_win = self.idata.posterior['P_win'].mean(dim=['chain', 'draw'])
        p_win_cf = self.idata.posterior['P_win_cf'].mean(dim=['chain', 'draw'])
        WARP = (p_win - p_win_cf).rename('WARP')
        return from_xarray(WARP)

    def _set_quantiles(
        self,
        cred_levels: Union[List[float], float]
    ) -> list:

        quantiles = [0.5]
        levels = [cred_levels] if isinstance(cred_levels, float) else cred_levels
        for level in levels:
            lower = (1 - level) / 2
            upper = level + lower
            quantiles.append(lower)
            quantiles.append(upper)

        return quantiles

    def _write_parameter_summary(
        self,
        posterior: Dataset,
        file: str,
        variable: str,
        cred_levels: Union[List[float], float] = [0.66, 0.8, 0.95]
    ):

        quantiles = self._set_quantiles(cred_levels)
        summary = posterior[variable].quantile(q=quantiles, dim=['chain', 'draw'])
        from_xarray(summary).write_parquet(file=file)

    def _extract_idata(
        self
    ) -> InferenceData:

        war_data = self.war_fit.war_data
        model_data = war_data.prepped_data
        full_data = war_data.full_data
        cids = war_data.cids

        cols = model_data.columns
        exclusions = [
            'cycle', 'state_name', 'district', 'pct', 'candidate_DEM', 'candidate_REP',
            'cid_DEM', 'cid_REP', 'eid'
        ]

        variables = [x for x in cols if x not in exclusions]

        # Dataset dimensions
        N = model_data.shape[0]
        M = full_data.shape[0]

        # Coordinates for candidates
        candidates = (
            cids
            .unique('cid')
            .with_columns(
                when(col.cid == 1)
                .then(lit('Generic Challenger'))
                .otherwise(col.candidate)
                .alias('candidate')
            )
            .sort('cid')
            ['candidate']
            .to_list()
        )

        # Coordinates for election cycle
        cycles = (
            model_data
            .unique('eid')
            .sort('cycle')
            ['cycle']
            .to_list()
        )

        # Coordinates for the posterior
        coords = {
            'N': range(N),
            'M': range(M),
            'cycle': cycles,
            'candidate': candidates,
            'variable': variables,
            'Em': cycles[1:],
            'party': ['dem', 'rep']
        }

        # Dimensions of each variable in terms of coordinate system
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
            'sigma_e': ['cycle'],

            # Transformed parameters
            'beta_c': ['candidate'],
            'alpha': ['cycle'],
            'beta_v': ['variable', 'cycle'],

            # Generated quantities
            'Y_rep': ['M'],
            'Y_rep_cf': ['party', 'M'],
            'P_win': ['party', 'M'],
            'P_win_cf': ['party', 'M'],
            'WAR': ['party', 'M']
        }

        idata = from_cmdstanpy(
            posterior=self.war_fit.war_fit,
            coords=coords,
            dims=dims
        )

        return idata