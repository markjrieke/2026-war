from typing import List, Union
from os.path import join, exists

from arviz import InferenceData, from_cmdstanpy
from polars import DataFrame, col, implode, lit, when, read_parquet, len
from polars.selectors import exclude
from plotnine import ggplot, aes, geom_pointrange, geom_abline, facet_wrap
from xarray import Dataset

from war.model import WARModel
from war.utils.transformations import from_xarray

class WARResults:

    def __init__(
        self,
        war_fit: WARModel
    ):

        """
        Class and methods for writing results of a model fit to disk.

        In addition to Wins Above Replacement (WAR), estimated by the model
        directly (see the `WARModel` class for more information), `WARResults`
        uses the posterior to estimate a new metric, WARP (Wins Above Replacement
        in terms of Probability of winning). For each candidate in each race,
        WARP is the candidate's predicted probability of winning minus the
        predicted probability of a hypothetical replacement candidate winning.

        A positive WARP implies that a replacement candidate has a lower
        probability of winning the seat and a negative WARP implies the inverse.
        WARP is a function of both the candidate quality and the competitiveness
        of the district. A candidate in a safe seat can have a relatively large
        WAR (either positive or negative) and still have a WARP of 0, whereas a
        candidates in hyper-competitive districts can have large WARPs from
        relatively small WARs.

        Parameters
        ----------
        war_fit : WARModel
            A fitted `WARModel` object.
        """

        self.war_fit = war_fit
        self.idata = self._extract_idata()

    def write_full_topline(
        self,
        path: str = 'out/summary',
        cred_level: float = 0.9
    ):

        """
        Write the topline WAR / WARP results for each congressional
        representative from 2000-2024, based on the results of November elections.

        The results are saved as `full_topline.parquet`.

        Parameters
        ----------
        path : str
            The path where the results topline results will be stored.
        cred_level : float
            Determines the size of the equal-tail-intervals (ETIs) for
            summarizing candidate WAR.
        """

        self._extract_topline(cred_level).write_parquet(join(path, 'full_topline.parquet'))

    def write_publication_topline(
        self,
        path: str = 'out/summary',
        cred_level: float = 0.9
    ):

        """
        Write the topline WAR / WARP results for congressional representatives
        elected in November 2024.

        The results are saved as `current_topline.csv`.

        Parameters
        ----------
        path : str
            The path where the topline results will be stored.
        cred_level : float
            Determines the size of the equal-tail-intervals (ETIs) for
            summarizing candidate WAR.
        """

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

        """
        Write out summary DataFrames for all named parameters in the model.

        Summary DataFrames are stored in long format with columns for the
        summary quantile, any parameter dimensions, and the parameter value
        at the summary quantile.

        Parameters
        ----------
        path : str
            The path where the parameter summaries will be stored.
        cred_levels: Union[List[float], float]
            The value (or list of values) determining the size of the
            equal-tail-intervals (ETIs) for summarizing parameter values.
        """

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

    def plot_holdout(
        self,
        cred_level: float = 0.95
    ) -> ggplot:

        """
        Generate a posterior predictive plot for the holdout set.

        Parameters
        ----------
        cred_level : float
            The value determining the size of the equal-tail-intervals (ETIs)
            displayed in the plot.
        """

        if not self.war_fit.holdout:
            raise ValueError(
                'This model was fit without a holdout set! Cannot generate a holdout plot.'
            )

        # Summarize Y_rep predictions
        quantiles = self._set_quantiles(cred_level)
        summary = (
            self.idata.posterior['Y_rep'].quantile(q=quantiles, dim=['chain', 'draw'])
        )
        summary = from_xarray(summary)

        summary = (
            summary
            .with_columns(col.quantile.round(decimals=3))
        )

        plot_quantiles = summary.unique('quantile').sort('quantile')['quantile']

        out = (
            summary
            .filter(col.quantile.is_in(plot_quantiles))
            .pivot(on='quantile', values='Y_rep')
            .join(
                self.war_fit.full_data,
                on='M',
                how='left'
            )
            .filter(
                col.uncontested == 0,
                col.cycle == self.war_fit.holdout
            )
            >>
            ggplot(aes(
                x='pct',
                y=str(plot_quantiles[1]),
                ymin=str(plot_quantiles[0]),
                ymax=str(plot_quantiles[2])
            )) +
            geom_pointrange(alpha=0.125) +
            geom_abline(linetype='dashed', color='red') +
            facet_wrap(facets='cycle')
        )

        return out

    def write_fit_summary(
        self,
        path: str = 'out/holdout',
        reset: bool = False
    ):

        """
        Write fit diagnostic summary information

        Model fit(s) are summarized using the posterior median RMSE for both the
        training data and holdout observations.

        Parameters
        ----------
        path : str
            The path where the fit summary will be stored.
        reset : bool
            Whether (`True`) or not (`False`) to overwrite the existing fit
            summary file.
        """

        if not self.war_fit.holdout:
            raise ValueError(
                'This model was fit without a holdout set! Cannot generate a holdout plot.'
            )

        predictions = (
            self.idata.posterior['Y_rep'].quantile(q=0.5, dim=['chain', 'draw'])
        )
        predictions = from_xarray(predictions)

        fit_summary = (
            predictions
            .pivot(on='quantile', values='Y_rep')
            .rename({'0.5': 'Y_rep'})
            .join(
                self.war_fit.full_data,
                on='M',
                how='left'
            )
            .filter(col.uncontested == 0)
            .with_columns(
                when(col.cycle == self.war_fit.holdout)
                .then(lit('holdout'))
                .otherwise(lit('training'))
                .alias('set')
            )
            .group_by('set')
            .agg(
                (((col.Y_rep - col.pct).pow(2)).sum() / len()).pow(0.5).alias('rmse')
            )
            .with_columns(lit(self.war_fit.holdout).alias('holdout_year'))
        )

        filepath = join(path, 'fit_summary.parquet')
        if reset or not exists(filepath):
            fit_summary.write_parquet(filepath)
        else:
            (
                fit_summary
                .vstack(read_parquet(filepath))
                .write_parquet(filepath)
            )


    def write_mappings(
        self,
        path: str = 'out/summary/mappings'
    ):

        """
        Write the set of DataFrames that map year, state, district, and
        candidates to indices found in the parameter summary DataFrames `M`
        maps row indices to the full dataset including uncontested races, `N`
        maps row indices to the subset used for model fitting.

        Parameters
        ----------
        path : str
            The path where the mapping tables will be stored.
        """

        cols = [
            'cycle', 'state_name', 'district', 'pct', 'uncontested',
            'candidate_DEM', 'candidate_REP', 'cid_DEM', 'cid_REP'
        ]

        (
            self.war_fit.full_data
            .select(['M'] + cols)
            .write_parquet(join(path, 'full_data.parquet'))
        )

        cols.remove('uncontested')

        (
            self.war_fit.model_data
            .select(['N'] + cols)
            .write_parquet(join(path, 'model_data.parquet'))
        )

    def _extract_topline(
        self,
        cred_level: float
    ) -> DataFrame:

        """
        Util method for converting topline results (WAR, WARP) for each
        representative to a DataFrame.
        """

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
        
        """Internal method for calculating WARP from posterior draws."""

        p_win = self.idata.posterior['P_win'].mean(dim=['chain', 'draw'])
        p_win_cf = self.idata.posterior['P_win_cf'].mean(dim=['chain', 'draw'])
        WARP = (p_win - p_win_cf).rename('WARP')
        return from_xarray(WARP)

    def _set_quantiles(
        self,
        cred_levels: Union[List[float], float]
    ) -> list:

        """
        Internal method for generating a list of quantiles based on a list of
        cred levels (or single cred level value)
        """

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

        """
        Internal method for writing the summary DataFrame for a single parameter.
        """

        quantiles = self._set_quantiles(cred_levels)
        if 'P_win' in variable:
            summary = posterior[variable].mean(dim=['chain', 'draw'])
        else:
            summary = posterior[variable].quantile(q=quantiles, dim=['chain', 'draw'])
        from_xarray(summary).write_parquet(file=file)

    def _extract_idata(
        self
    ) -> InferenceData:

        """
        Internal method for extracting an InferenceData object with named
        dimension and coordinates from a fitted CmdStanMCMC object.
        """

        war_data = self.war_fit.war_data
        time_varying_variables = self.war_fit.time_varying_variables
        time_invariant_variables = self.war_fit.time_invariant_variables
        sd_variables = self.war_fit.sd_variables
        fec_variables = self.war_fit.fec_variables

        model_data = war_data.prepped_data
        full_data = war_data.full_data
        cids = war_data.cids

        if self.war_fit.holdout:
            model_data = model_data.filter(col.cycle < self.war_fit.holdout)
            full_data = full_data.filter(col.cycle <= self.war_fit.holdout)

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
            full_data
            .unique('eid')
            .sort('cycle')
            ['cycle']
            .to_list()
        )

        # Coordinates for the posterior
        coords = {
            'N': range(N),
            'M': range(M),
            'Em': cycles[1:],
            'cycle': cycles,
            'candidate': candidates,
            'time_varying_variable': time_varying_variables,
            'time_invariant_variable': time_invariant_variables,
            'sd_variable': sd_variables,
            'fec_variable': fec_variables,
            'party': ['dem', 'rep']
        }

        # Dimensions of each variable in terms of coordinate system
        dims = {
            # Global hyper-parameter
            'sigma': [],

            # Variable scale parameter offsets
            'eta_sigma_alpha': [],
            'eta_sigma_beta_d': ['time_varying_variable'],
            'eta_sigma_beta_g': [],
            'eta_sigma_beta_j': [],
            'eta_sigma_beta_c': [],
            'eta_sigma_sigma_e': [],
            'eta_sigma_theta_f': [],
            'esta_sigma_alpha_f': [],
            'eta_sigma_beta_f': [],
            'eta_sigma_sigma_f': [],
            'eta_sigma_theta_e': ['party'],

            # Random walk initial states
            'eta_alpha0': [],
            'eta_beta_d0': ['time_varying_variable'],
            'eta_sigma_e0': [],
            'eta_theta_f0': [],
            'eta_alpha_f0': [],
            'eta_sigma_f0': [],
            'eta_theta_e0': ['party'],

            # Variable scale parameter offsets
            'eta_alpha': ['Em'],
            'eta_beta_d': ['time_varying_variable', 'Em'],
            'eta_beta_g': ['time_invariant_variable'],
            'eta_beta_j': ['sd_variable'],
            'eta_beta_c': ['candidate'],
            'eta_sigma_e': ['Em'],
            'eta_theta_f': ['Em'],
            'eta_alpha_f': ['Em'],
            'eta_beta_f': ['fec_variable'],
            'eta_sigma_f': ['Em'],
            'eta_theta_e': ['party', 'Em'],

            # Transformed parameters
            'beta_c': ['candidate'],
            'beta_g': ['time_invariant_variable'],
            'beta_j': ['sd_variable'],
            'alpha': ['cycle'],
            'beta_d': ['time_varying_variable', 'cycle'],
            'sigma_e': ['cycle'],
            'theta_f': ['cycle'],
            'alpha_f': ['cycle'],
            'beta_f': ['fec_variable'],
            'sigma_f': ['cycle'],
            'theta_e': ['party', 'cycle'],

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