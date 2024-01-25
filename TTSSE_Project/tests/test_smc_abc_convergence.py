import numpy as np
import pytest
from TTSSE_Project.utilities.priors.uniform_prior import UniformPrior
from TTSSE_Project.utilities.distancefunctions.euclidean import EuclideanDistance
from TTSSE_Project.utilities.summarystats import auto_tsfresh
from TTSSE_Project.inference.smc_abc import SMCABC
from TTSSE_Project.utilities.epsilonselectors.relative_epsilon_selector import RelativeEpsilonSelector

# MA2 Simulator
def simulator(param, n=100):
    m = len(param)
    g = np.random.normal(0, 1, n)
    gy = np.random.normal(0, 0.3, n)
    y = np.zeros(n)
    x = np.zeros(n)
    for t in range(0, n):
        x[t] += g[t]
        for p in range(0, np.minimum(t, m)):
            x[t] += g[t - 1 - p] * param[p]
        y[t] = x[t] + gy[t]
    
    return np.reshape(y, (1,1,100))  # NxSxT format

# Generate observed data
obs_data = simulator([0.6,0.2])

# Define priors
parameter_names = ['k1', 'k2']
lower_bounds = [0, 0]
upper_bounds = [1, 1]
prior = UniformPrior(np.array(lower_bounds), np.array(upper_bounds))


values = np.array([1.0, 1.0])

parameters = {}
for i, name in enumerate(parameter_names):
    parameters[name] = values[i]

# Define distance function
distance_func = EuclideanDistance()

# Define summary statistics
default_fc_params = {
                     'variance':None,
                     'autocorrelation':
                         [{'lag':1},
                          {'lag':2}]}

summaries = auto_tsfresh.SummariesTSFRESH(features=default_fc_params)

def run_smcabc_inference():
    max_rounds = 3
    eps_selector = RelativeEpsilonSelector(20, max_rounds)
    
    smcabc = SMCABC(obs_data,  # Observed Dataset
              simulator, # Simulator method
              prior, # Prior
              summaries_function=summaries.compute)

    smc_abc_results = smcabc.infer(num_samples = 10, batch_size = 1,
                                   chunk_size=1, eps_selector=eps_selector)
    
    
    
    #inferred_parameters = smc_abc_results[-1]["accepted_samples"]
    return smc_abc_results

# Test the convergence for smc-abc
# If the values of eps is decreasing in subsequent rounds then the algorithm is said to be converging

def test_smcabc_convergence():
    # Run the SMC-ABC inference
    smc_abc_results = run_smcabc_inference()
    print(smc_abc_results)
    
    # Extract epsilon values used in the inference
    eps_values = [res["eps"] for res in smc_abc_results]
    
    # Ensure that epsilon values are in a decreasing order
    for i in range(1, len(eps_values)):
        assert eps_values[i] <= eps_values[i-1], f"Epsilon value increased from iteration {i-1} to {i}"

test_smcabc_convergence()


