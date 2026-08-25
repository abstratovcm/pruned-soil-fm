"""Canonical repository paths, anchored to the repository root.

Every script and notebook resolves input and output locations through this
module.

Layout::

    data/prepared_datasets/<variant>/<variant>.csv   inputs (not redistributable)
    results/flow_matching/<variant>/                 ours
    results/tabsyn/<variant>/                        baseline
    results/downstream/<variant>/                    PLSR utility
    results/tables/                                  generated LaTeX
    results/figures/                                 generated figures
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Inputs
DATA_DIR = os.path.join(ROOT, "data")
PREPARED_DATA = os.path.join(DATA_DIR, "prepared_datasets")

# Vendored baseline (cloned separately, see README)
TABSYN_DIR = os.path.join(ROOT, "tabsyn")

# Outputs
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_FM = os.path.join(RESULTS_DIR, "flow_matching")
RESULTS_TABSYN = os.path.join(RESULTS_DIR, "tabsyn")
RESULTS_DOWNSTREAM = os.path.join(RESULTS_DIR, "downstream")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def prepared_dataset(variant):
    """Path to the prepared CSV for a dataset variant."""
    return os.path.join(PREPARED_DATA, variant, f"{variant}.csv")


def fm_results(variant):
    """Flow Matching results directory for a dataset variant."""
    return os.path.join(RESULTS_FM, variant)


def tabsyn_results(variant):
    """TabSyn baseline results directory for a dataset variant."""
    return os.path.join(RESULTS_TABSYN, variant)


def downstream_results(variant):
    """Downstream (PLSR) results directory for a dataset variant."""
    return os.path.join(RESULTS_DOWNSTREAM, variant)


def tabsyn_data(variant):
    """Working directory TabSyn reads/writes inside the vendored clone."""
    return os.path.join(TABSYN_DIR, "data", variant)
