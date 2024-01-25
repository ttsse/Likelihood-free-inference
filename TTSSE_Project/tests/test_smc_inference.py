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

parameters_ = {}
for i, name in enumerate(parameter_names):
    parameters_[name] = values[i]

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
              summaries_function=summaries.compute
              )

    smc_abc_results = smcabc.infer(num_samples = 10, batch_size = 1,
                                   chunk_size=1, eps_selector=eps_selector
                                   )
    
    
    
    inferred_parameters = smc_abc_results[-1]["accepted_samples"]
    return inferred_parameters

def test_smcabc_inference():
    inferred_params = run_smcabc_inference()
    
    # There should be 10 accepted samples after the inference
    assert len(inferred_params) == 10, "SMC-ABC Inference did not return the expected number of samples"

    




