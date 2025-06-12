import arviz as az

from war.data import WARData
from war.model import WARModel

war_data = WARData().prep_data()
war_fit = (
    WARModel(
        data=war_data,
        stan_file='stan/dev_01.stan',
        dir='exe'
    )
    .sample(
        iter_warmup=1000,
        iter_sampling=1000,
        chains=8,
        parallel_chains=8
    )
)

# TODO blegh ----
az.from_cmdstanpy(posterior=war_fit).posterior