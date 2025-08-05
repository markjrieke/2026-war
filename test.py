from polars import read_parquet

from argparse import ArgumentParser

from war.data import WARData
from war.model import WARModel
from war.results import WARResults

parser = ArgumentParser(
    prog='Out-of-sample Fit Estimates',
    description='Evaluate out-of-sample model fit on holdout data'
)

parser.add_argument('--chamber', choices=['house', 'senate'], required=True)
parser.add_argument('--iter_warmup', type=int, default='100')
parser.add_argument('--iter_sampling', type=int, default='100')
parser.add_argument('--chains', type=int, default='10')
parser.add_argument('--parallel_chains', type=int, default='10')

args = parser.parse_args()

war_data = WARData(args.chamber).prep_data()

for holdout_cycle in [2018, 2020, 2022, 2024]:
    war_fit = (
        WARModel(
            war_data=war_data,
            stan_file='stan/war.stan',
            dir='exe'
        )
        .prep_stan_data(holdout=holdout_cycle)
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

    # Write holdout results summaries/plots
    reset = holdout_cycle == 2018
    war_results = WARResults(war_fit)
    war_results.plot_holdout().save(f'out/holdout/{args.chamber}_plots/{holdout_cycle}.png')
    war_results.write_fit_summary(reset=reset)

