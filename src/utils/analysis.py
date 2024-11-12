# Copyright (c) 2024 Niko Natsoulas
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.stats import chi2
from typing import Dict

def perform_nees_hypothesis_test(nees_values: np.ndarray, alpha: float = 0.05, n_states: int = 6) -> Dict:
    """
    Perform NEES Chi-square hypothesis test
    
    Args:
        nees_values: Array of NEES values
        alpha: Significance level (default 0.05 for 95% confidence)
        n_states: Number of states (default 6 for our system)
        
    Returns:
        Dictionary containing test results
    """
    # Remove any NaN values
    valid_nees = nees_values[~np.isnan(nees_values)]
    N = len(valid_nees)
    
    # Compute average NEES
    avg_nees = np.mean(valid_nees)
    
    # Chi-square test bounds for n*N degrees of freedom
    # Where n is number of states and N is number of samples
    dof = n_states
    
    # Lower and upper bounds for individual NEES values
    r1 = chi2.ppf(alpha/2, dof)
    r2 = chi2.ppf(1 - alpha/2, dof)
    
    # Bounds for average NEES
    r1_bar = r1/N
    r2_bar = r2/N
    
    # Percentage of samples within bounds
    within_bounds = np.mean((valid_nees >= r1) & (valid_nees <= r2)) * 100
    
    # Test results
    results = {
        'filter_consistent': r1 <= avg_nees <= r2,
        'average_nees': avg_nees,
        'lower_bound': r1,
        'upper_bound': r2,
        'percent_in_bounds': within_bounds,
        'n_samples': N
    }
    
    return results 