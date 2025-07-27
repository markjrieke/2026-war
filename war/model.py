from typing import Optional

from cmdstanpy import CmdStanMCMC

from war.data import WARData
from war.utils.model import CmdStanModel
from war.utils.constants import DEFAULT_PRIORS

class WARModel:

    def __init__(
        self,
        war_data: WARData,
        stan_file: str,
        **kwargs
    ):
        self.war_data = war_data
        self.war_model = CmdStanModel(stan_file=stan_file, **kwargs)

    def prep_stan_data(
        self,
        priors: Optional[dict] = None,
        prior_check: bool = False
    ):
        model_data = self.war_data.prepped_data
        full_data = self.war_data.full_data
        cids = self.war_data.cids

        # Extract variable names from the model frame
        cols = model_data.columns
        exclusions = [
            'cycle', 'state_name', 'district', 'pct', 'candidate_DEM', 'candidate_REP',
            'cid_DEM', 'cid_REP', 'eid'
        ]
        variables = [x for x in cols if x not in exclusions]

        # Find locations of incumbent ID columns
        iid = [
            variables.index('is_incumbent_DEM') + 1,
            variables.index('is_incumbent_REP') + 1
        ]

        # Parse stan data from model frame
        stan_data = {
            'N': model_data.shape[0],
            'M': full_data.shape[0],
            'E': model_data.unique('eid').shape[0],
            'C': cids.unique('cid').shape[0],
            'F': len(variables),
            'X': model_data.select(variables).to_numpy(),
            'Y': model_data['pct'].to_numpy(),
            'Xf': full_data.select(variables).to_numpy(),
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

        return self

    def sample(
        self,
        **kwargs
    ):

        self.war_fit = self.war_model.sample(
            data=self.stan_data,
            **kwargs
        )

        return self
