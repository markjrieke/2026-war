from cmdstanpy import CmdStanMCMC
from polars.selectors import exclude
from numpy import repeat

from war.utils.model import CmdStanModel
from war.data import WARData

class WARModel:

    # TODO model config options
    def __init__(
        self,
        data: WARData,
        stan_file: str,
        **kwargs
    ):
        self.prepped_data = data.prepped_data
        self.war_model = CmdStanModel(stan_file=stan_file, **kwargs)

    # TODO data prep based on config
    def sample(
        self,
        **kwargs
    ) -> CmdStanMCMC:

        # Convert prepped data to stan_data
        Y = self.prepped_data['margin'].to_numpy()
        X = self.prepped_data.select(exclude('margin')).to_numpy()
        N, K = X.shape

        stan_data = {
            'N': N,
            'K': K,
            'X': X,
            'Y': Y,
            'alpha_mu': 0,
            'alpha_sigma': 1,
            'beta_mu': repeat(0, K),
            'beta_sigma': repeat(1, K),
            'sigma_sigma': 1
        }

        war_fit = self.war_model.sample(
            data=stan_data,
            **kwargs
        )

        return war_fit
