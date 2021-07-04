from pathlib import Path

from .timestamp import timestamp

RESULTS_DIR_PREFIX = 'results_'


def make_results_dir(results_root, prefix=RESULTS_DIR_PREFIX):
    """make a directory to contain results from an experiment
    within a specified "root" results directory

    Parameters
    ----------
    results_root : str, pathlib.Path
        root directory where results directories should be made

    Returns
    -------
    results_dir_path : pathlib.Path
    """
    results_root = Path(results_root)
    if not results_root.is_dir():
        raise NotADirectoryError(
            f'path specified for results_root not found: {results_root}'
        )
    results_dir_path = results_root.joinpath(
        f'{prefix}{timestamp()}'
    )
    results_dir_path.mkdir()
    return results_dir_path
