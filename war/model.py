from typing import Optional, List

from polars import col

from war.data import WARData
from war.utils.model import CmdStanModel
from war.utils.constants import (
    DEFAULT_PRIORS,
    TIME_VARYING_VARIABLES,
    TIME_INVARIANT_VARIABLES,
    SD_VARIABLES
)

class WARModel:

    def __init__(
        self,
        war_data: WARData,
        stan_file: str,
        **kwargs
    ):

        """
        Class for fitting a hierarchical Bayesian model to estimate each
        candidate's Wins Above Replacement (WAR) metric.

        This model estimates candidate "skill" as a model parameter directly.
        Each candidate's WAR is then the difference in potential outcomes
        between the predictive distribution and the predictive distribution
        with a hypothetical new candidate.

        Parameters
        ----------
        war_data : WARData
            A prepped `WARData` object.
        stan_file : str
            The path to the model file.
        **kwargs
            Other named arguments to pass to `CmdStanModel`. This class is the
            wrapper class found under `war.utils.model`, which contains an
            additional optional argument, `dir`, for specifying where the Stan
            executable should be created (in addition to all the named arguments
            in the cmdstanpy class of the same name).
        """

        self.war_data = war_data
        self.war_model = CmdStanModel(stan_file=stan_file, **kwargs)

    def prep_stan_data(
        self,
        priors: Optional[dict] = None,
        prior_check: bool = False
    ):

        """
        Attatch a dictionary containing data for passing to the model.

        Parameters
        ----------
        priors : Optional[dict]
            An optional dictionary containing priors for the model. If `None`,
            the default priors found under `war.utils.constants` will be used.
        prior_check : bool
            If `True`, the model will sample from the prior predictive
            distribution rather than fit to the data.
        """

        model_data = self.war_data.prepped_data
        full_data = self.war_data.full_data
        cids = self.war_data.cids

        # Extract variable names from the model frame
        cols = model_data.columns
        time_varying_variables = [x for x in cols if x in TIME_VARYING_VARIABLES]
        time_invariant_variables = [x for x in cols if x in TIME_INVARIANT_VARIABLES]
        sd_variables = [x for x in cols if x in SD_VARIABLES]

        # Find locations of incumbent ID columns
        iid = [
            time_varying_variables.index('is_incumbent_DEM') + 1,
            time_varying_variables.index('is_incumbent_REP') + 1
        ]

        # Parse stan data from model frame
        stan_data = {
            'N': model_data.shape[0],
            'M': full_data.shape[0],
            'E': model_data.unique('eid').shape[0],
            'C': cids.unique('cid').shape[0],
            'D': len(time_varying_variables),
            'L': len(time_invariant_variables),
            'J': len(sd_variables),
            'Xd': model_data.select(time_varying_variables).to_numpy(),
            'Xl': model_data.select(time_invariant_variables).to_numpy(),
            'Xj': model_data.select(sd_variables).to_numpy(),
            'Y': model_data['pct'].to_numpy(),
            'Xfd': full_data.select(time_varying_variables).to_numpy(),
            'Xfl': full_data.select(time_invariant_variables).to_numpy(),
            'Xfj': full_data.select(sd_variables).to_numpy(),
            'cid': model_data['cid_DEM', 'cid_REP'].to_numpy(),
            'eid': model_data['eid'].to_numpy(),
            'cfid': full_data['cid_DEM', 'cid_REP'].to_numpy(),
            'efid': full_data['eid'].to_numpy(),
            'iid': iid
        }

        # Add in priors if supplied, or use default priors
        priors_args = DEFAULT_PRIORS
        if priors:
            for key, value in priors:
                if key not in priors_args.keys():
                    raise KeyError(f'{key} is not a valid parameter!')
                priors_args.update({key: value})

        # Sample from the prior or fit to the data
        prior_check = 1 if prior_check else 0
        priors_args.update({'prior_check': prior_check})

        # Attach the full stan data object
        stan_data.update(priors_args)
        self.stan_data = stan_data
        self.time_varying_variables = time_varying_variables
        self.time_invariant_variables = ['intercept'] + time_invariant_variables
        self.sd_variables = sd_variables

        return self

    def sample(
        self,
        **kwargs
    ):

        """
        A wrapper around cmdstanpy's `sample()` method.

        Parameters
        ----------
        **kwargs
            Named arguments for cmdstanpy's `sample()` method, excluding `data`
            (this is passed to the method internally).
        """

        self.war_fit = self.war_model.sample(
            data=self.stan_data,
            **kwargs
        )

        return self

    def _detect_national_variables(
        self,
        exclusions
    ) -> List[str]:

        # Set list of columns to look through as potential national variables
        full_data = self.war_data.full_data
        columns = full_data.columns
        for exclusion in exclusions:
            columns.remove(exclusion)

        # Generate list of national variables (only vary by cycle)
        variables = []
        for column in columns:
            n = (
                full_data
                .group_by('cycle')
                .agg(col(column).n_unique())
                [column]
            )
            if (n == 1).all():
                variables.append(column)

        return variables