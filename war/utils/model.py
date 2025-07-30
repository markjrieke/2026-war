from typing import Optional
from os.path import basename, dirname, join
from os import listdir, remove
from shutil import copy

import cmdstanpy as stan

class CmdStanModel(stan.CmdStanModel):

    """
    A small wrapper class around cmdstanpy's `CmdStanModel` class that adds
    functionality for generating an executable in a new directory. While this is
    functionally similar to the `cmdstan_model()` function in R, it differs in
    that the stan file is *also* copied to the new directory.
    """

    def __init__(
        self,
        stan_file: str,
        dir: Optional[str] = None,
        **kwargs
    ):

        if dir:
            # Check if the dir file needs to be updated
            if basename(stan_file) in listdir(dir):
                source_lines = open(stan_file).readlines()
                destination_lines = open(join(dir, basename(stan_file))).readlines()
                if source_lines != destination_lines:
                    copy(stan_file, dir)
            else:
                copy(stan_file, dir)

            # Init CmdStanModel object with new file location reference
            model_dir = dirname(stan_file)
            stan_file = join(dir, basename(stan_file))

            # Append stanc_options with model_dir in include-paths
            if 'stanc_options' in kwargs.keys():
                stanc_options: dict = kwargs.get('stanc_options')
                if 'include-paths' in stanc_options.keys():
                    include_paths = stanc_options.get('include-paths')
                    if type(include_paths) == str:
                        kwargs['stanc_options'].update({'include-paths': [include_paths, model_dir]})
                    elif type(include_paths) == list:
                        kwargs['stanc_options']['include-paths'].append(model_dir)
                    else:
                        raise TypeError('include-paths must be of type str or list.')
                else:
                    kwargs['stanc_options'].update({'include-paths': model_dir})
            else:
                kwargs.update({'stanc_options': {'include-paths': model_dir}})

        # Init the CmdStanModel object
        super().__init__(
            stan_file=stan_file,
            **kwargs
        )

def clean_dir(dir: str = 'exe'):

    """Util function for removing all files in a directory"""

    files = listdir(dir)
    for f in files:
        remove(join(dir, f))
    print(f'Removed {len(files)} files from {dir}:')
    print('\n'.join(files))