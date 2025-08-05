from argparse import ArgumentParser

from war.data import WARData
from war.model import WARModel
from war.results import WARResults

parser = ArgumentParser(
    prog='Bayesian WAR',
    description='Estimate candidate WAR via hierarchical Bayesian model'
)

# Command line arguments for sampling
parser.add_argument('--chamber', choices=['house', 'senate'], required=True)
parser.add_argument('--iter_warmup', type=int, default='1000')
parser.add_argument('--iter_sampling', type=int, default='1000')
parser.add_argument('--chains', type=int, default='10')
parser.add_argument('--parallel_chains', type=int, default='10')

args = parser.parse_args()

# Import and prep data for running the model
war_data = WARData(args.chamber).prep_data()

# Fit the model
war_fit = (
    WARModel(
        war_data=war_data,
        stan_file='stan/war.stan',
        dir='exe'
    )
    .prep_stan_data()
    .sample(
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        chains=args.chains,
        parallel_chains=args.parallel_chains,
        inits=0.01,
        step_size=0.002,
        refresh=20,
        seed=2026
    )
)

# Write results to out/
war_results = WARResults(war_fit)
war_results.write_full_topline()
war_results.write_publication_topline()
war_results.write_parameter_summaries()
war_results.write_mappings()