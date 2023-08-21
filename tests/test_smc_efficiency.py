import numpy as np
import sciope
import pytest
from sciope.utilities.priors.uniform_prior import UniformPrior
from sciope.utilities.distancefunctions.euclidean import EuclideanDistance
from sciope.utilities.summarystats import auto_tsfresh
from sciope.inference.smc_abc import SMCABC
from sciope.visualize.inference_results import InferenceResults
from sciope.utilities.epsilonselectors.relative_epsilon_selector import RelativeEpsilonSelector


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
    
    return np.reshape(y, (1,1,100))

# Generate observed data
obs_data = simulator([0.6,0.2])

# Define priors
parameter_names = ['k1', 'k2']
lower_bounds = [0, 0]
upper_bounds = [1, 1]
prior = UniformPrior(np.array(lower_bounds), np.array(upper_bounds))


values = np.array([
    1.0, 1.0
])

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
    max_rounds = 6
    eps_selector = RelativeEpsilonSelector(20, max_rounds)
    
    smcabc = SMCABC(obs_data,  # Observed Dataset
              simulator, # Simulator method
              prior, # Prior
              summaries_function=summaries.compute,
              parameters =  parameters)

    smc_abc_results = smcabc.infer(num_samples = 10, batch_size = 1,
                                   chunk_size=1, eps_selector=eps_selector, resume = False)
    return smc_abc_results

def test_smcabc_efficiency():
    
    smc_abc_results = run_smcabc_inference()

    # efficiency = (accepted_sample_per_round) / (total_num_trials_per_round)
    
    desired_num_accepted = 10  # For each round we want 10 accpeted samples
    efficiencies = [desired_num_accepted / res["trial_count"] for res in smc_abc_results]
    trial_counts = [res["trial_count"] for res in smc_abc_results]
    
    # Ensure that efficiency values are non-increasing and positive
    for i in range(1, len(efficiencies)):
        assert efficiencies[i] <= efficiencies[i-1] + 1e-10, f"Efficiency significantly increased from round {i-1} to {i}"
        assert efficiencies[i] > 0, f"Efficiency is non-positive in round {i}"


