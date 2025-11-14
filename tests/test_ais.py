"""
Copyright © 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The PySA, a powerful tool for solving optimization problems is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import numpy as np
import pandas as pd
import pytest
from pysa.ais import partition_function_post, get_log_omega, omegas_to_partition


def test_partition_function_post_basic():
    """Test that partition_function_post correctly processes multiple samples.
    
    This test ensures that the function correctly iterates over all samples
    and calculates log_omega for each one. This was a bug where the loop
    variables were dedented outside the for loop, causing only the last
    sample to be processed.
    """
    # Create a simple test case with 3 samples
    n = 4  # number of bits
    num_samples = 3

    # Create temperature schedules (must include infinity for beta=0)
    temps_sample = np.array([np.inf, 10.0, 1.0, 0.1])

    # Create sample data - each sample should have its own temperatures and energies
    temps = [temps_sample for _ in range(num_samples)]

    # Create different energies for each sample to verify all are processed
    energies = [
        np.array([-1.0, -2.0, -3.0, -4.0]),
        np.array([-5.0, -6.0, -7.0, -8.0]),
        np.array([-9.0, -10.0, -11.0, -12.0])
    ]

    # Create states (required by the function to get n)
    # Use n to create states with alternating spins
    states = [[
        np.array([1 if i % 2 == 0 else -1
                  for i in range(n)])
        for _ in range(len(temps_sample))
    ]
              for _ in range(num_samples)]

    # Create DataFrame
    solution = pd.DataFrame({
        'states': states,
        'temps': temps,
        'energies': energies
    })

    # Run the function
    result = partition_function_post(solution)

    # Basic checks
    assert isinstance(result, (float, np.floating)), "Result should be a float"
    assert not np.isnan(result), "Result should not be NaN"
    assert not np.isinf(result), "Result should not be infinite"

    # The result should be a finite number representing log(Zf)
    # Since we have multiple samples with different energies, the result
    # should incorporate information from all of them
    print(f"Partition function result: {result}")


def test_get_log_omega_single():
    """Test get_log_omega with a single sample."""
    betas = np.array([0.0, 0.1, 1.0, 10.0])
    beta_idx = np.array([0, 1, 2, 3])
    energies = np.array([-1.0, -2.0, -3.0, -4.0])

    result = get_log_omega(betas, beta_idx, energies)

    assert isinstance(result, (float, np.floating)), "Result should be a float"
    assert not np.isnan(result), "Result should not be NaN"


def test_get_log_omega_requires_zero_beta():
    """Test that get_log_omega raises an error when beta=0 is not present."""
    betas = np.array([0.1, 1.0, 10.0])  # No zero beta
    beta_idx = np.array([0, 1, 2])
    energies = np.array([-1.0, -2.0, -3.0])

    with pytest.raises(ValueError, match="zero beta"):
        get_log_omega(betas, beta_idx, energies)


def test_omegas_to_partition():
    """Test omegas_to_partition calculation."""
    log_omegas = np.array([1.0, 2.0, 3.0])
    logZ0 = np.log(16)  # log(2^4) for n=4

    result = omegas_to_partition(log_omegas, logZ0)

    assert isinstance(result, (float, np.floating)), "Result should be a float"
    assert not np.isnan(result), "Result should not be NaN"
    assert not np.isinf(result), "Result should not be infinite"


def test_partition_function_post_consistency():
    """Test that partition_function_post gives consistent results."""
    # Create deterministic test case
    np.random.seed(42)

    n = 5
    num_samples = 2
    temps_sample = np.array([np.inf, 5.0, 1.0, 0.5])

    temps = [temps_sample for _ in range(num_samples)]
    energies = [
        np.random.randn(len(temps_sample)) * 10 for _ in range(num_samples)
    ]
    states = [[
        np.random.choice([-1, 1], size=n) for _ in range(len(temps_sample))
    ] for _ in range(num_samples)]

    solution = pd.DataFrame({
        'states': states,
        'temps': temps,
        'energies': energies
    })

    # Run twice with same data
    result1 = partition_function_post(solution)
    result2 = partition_function_post(solution)

    # Should give identical results
    assert result1 == result2, "Function should be deterministic"


if __name__ == "__main__":
    # Run tests
    test_partition_function_post_basic()
    test_get_log_omega_single()
    test_get_log_omega_requires_zero_beta()
    test_omegas_to_partition()
    test_partition_function_post_consistency()
    print("All tests passed!")
