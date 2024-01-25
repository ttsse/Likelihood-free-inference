import numpy as np
from TTSSE_Project.features import feature_extraction as fe
from TTSSE_Project.utilities.priors import uniform_prior
from TTSSE_Project.inference.abc_inference import ABC
from TTSSE_Project.utilities.distancefunctions import naive_squared, manhattan,euclidean
from tsfresh.feature_extraction.settings import MinimalFCParameters
from sklearn.metrics import mean_absolute_error
from gillespy2.solvers.numpy import NumPySSASolver
from dask.distributed import Client
import gillespy2
import pytest
import numpy as np
from gillespy2 import SSACSolver

class ToggleSwitch(gillespy2.Model):
    """ Gardner et al. Nature (1999)
    'Construction of a genetic toggle switch in Escherichia coli'
    """

    def __init__(self, parameter_values=None):
        # Initialize the model.
        gillespy2.Model.__init__(self, name="toggle_switch")
        # Parameters
        alpha1 = gillespy2.Parameter(name='alpha1', expression=1)
        alpha2 = gillespy2.Parameter(name='alpha2', expression=1)
        beta = gillespy2.Parameter(name='beta', expression="2.0")
        gamma = gillespy2.Parameter(name='gamma', expression="2.0")
        mu = gillespy2.Parameter(name='mu', expression=1.0)
        self.add_parameter([alpha1, alpha2, beta, gamma, mu])

        # Species
        U = gillespy2.Species(name='U', initial_value=10)
        V = gillespy2.Species(name='V', initial_value=10)
        self.add_species([U, V])

        # Reactions
        cu = gillespy2.Reaction(name="r1", reactants={}, products={U: 1},
                                propensity_function="alpha1/(1+pow(V,beta))")
        cv = gillespy2.Reaction(name="r2", reactants={}, products={V: 1},
                                propensity_function="alpha2/(1+pow(U,gamma))")
        du = gillespy2.Reaction(name="r3", reactants={U: 1}, products={},
                                rate=mu)
        dv = gillespy2.Reaction(name="r4", reactants={V: 1}, products={},
                                rate=mu)
        self.add_reaction([cu, cv, du, dv])
        self.timespan(np.linspace(0, 50, 101))

toggle_model = ToggleSwitch()
solver = SSACSolver
parameter_names = ['alpha1', 'alpha2', 'beta', 'gamma', 'mu']

def simulator(params, model, transform=True):
    params = params.ravel()

    num_trajectories = 1  # TODO: howto handle ensembles
    res = model.run(
        solver=solver,
        variables={parameter_names[i]: params[i] for i in range(len(parameter_names))})

    if transform:

        U_ = res['U']
        V_ = res['V']

        return np.vstack([U_, V_])[np.newaxis, :, :]

    else:
        return res

def simulator2(x):
    return simulator(x, model=toggle_model)

default_param = np.array(list(toggle_model.listOfParameters.items()))[:, 1]
bound = []
for exp in default_param:
    bound.append(float(exp.expression))

true_params = np.array(bound)
dmin = true_params * 0.5
dmax = true_params * 2.0

# simulator2(true_params)

uni_prior = uniform_prior.UniformPrior(dmin, dmax)

fixed_data = toggle_model.run(solver=solver, number_of_trajectories=30)

results = []
for x in fixed_data:
    results.append(np.vstack([x['U'], x['V']])[np.newaxis, :, :])

results = np.array(results)

fixed_data = np.squeeze(results, axis=1)

summ_func = lambda x: fe.generate_tsfresh_features(x, MinimalFCParameters())

ns = naive_squared.NaiveSquaredDistance()
man = manhattan.ManhattanDistance()
eu = euclidean.EuclideanDistance()

import pytest
from dask.distributed import Client
from sklearn.metrics import mean_absolute_error
from TTSSE_Project.utilities.distancefunctions import naive_squared, manhattan, euclidean

DISTANCE_METRICS = [
    naive_squared.NaiveSquaredDistance(),
     euclidean.EuclideanDistance()
]

@pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
def test_abc_functional(distance_metric):
    abc = ABC(fixed_data, sim=simulator2, prior_function=uni_prior, summaries_function=summ_func, distance_function=distance_metric)

    abc.compute_fixed_mean(chunk_size=1)

    # run in multiprocessing mode
    res = abc.infer(num_samples=5, batch_size=1, chunk_size=1)
    mae_inference = mean_absolute_error(true_params, abc.results['inferred_parameters'])
    
    assert mae_inference < 0.5, f"ABC inference test failed with {distance_metric}, error too high"








