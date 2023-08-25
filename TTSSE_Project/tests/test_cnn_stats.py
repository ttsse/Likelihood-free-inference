
import numpy as np
import pytest

from TTSSE_Project.utilities.priors.uniform_prior import UniformPrior
from TTSSE_Project.models.cnn_regressor import CNNModel

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
    
    return np.reshape(y, (100))

# Generate observed data
obs_data = simulator([0.6,0.2])

# Define priors
parameter_names = ['k1', 'k2']
dmin = [0, 0]
dmax = [1, 1]


# In[3]:


# Set up simulation
# We use 10,000 samples in the training dataset to be able to run the notebook quickly, which is comparitively few
# For accurate inference, the number should be 1 or 2 orders of magnitude more.

n = 10000
train_thetas = np.random.uniform(low=dmin, high=dmax, size=(n,2))
train_ts = np.expand_dims(np.array([simulator(p, n=100) for p in train_thetas]), 2)

validation_thetas = np.random.uniform(low=dmin, high=dmax, size=(n,2))
validation_ts = np.expand_dims(np.array([simulator(p, n=100) for p in validation_thetas]), 2)

test_thetas = np.random.uniform(low=dmin, high=dmax, size=(n,2))
test_ts = np.expand_dims(np.array([simulator(p, n=100) for p in test_thetas]), 2)



# In[5]:


# Set up training data in the right format
train_ts = train_ts.transpose((0, 2, 1))
validation_ts = validation_ts.transpose((0, 2, 1))
test_ts = test_ts.transpose((0, 2, 1))
obs_data = obs_data.reshape(1,1,100)


# In[6]:


# Set input and output shape for the CNN
input_shape = (100,1)
output_shape = 2


# In[7]:


# Set up the search space for inference
dmin = [-2, -1]
dmax = [4, 2]


# In[8]:


# Routines to normalize and denormalize data
# Makes training easier
def normalize_data(data, dmin, dmax):
    dmin = np.array(dmin)
    dmax = np.array(dmax)
    return (data - dmin)/(dmax-dmin)

def denormalize_data(data, dmin, dmax):
    dmin = np.array(dmin)
    dmax = np.array(dmax)
    denorm = data * (dmax-dmin) + dmin
    return denorm


# In[9]:


normed_thetas = normalize_data(train_thetas, dmin, dmax)
normed_thetas_val = normalize_data(validation_thetas, dmin, dmax)


# In[10]:


# Instantiate the model
model_cnn = CNNModel(input_shape, output_shape)


# In[11]:


history_cnn = model_cnn.train(train_ts, normed_thetas, batch_size=256, 
                      epochs=20, verbose=0, learning_rate=0.001, 
                      early_stopping_patience=5,
                      validation_inputs=validation_ts, validation_targets=normed_thetas_val)


# In[15]:


def verify_model(model):
    pred_test = model.predict(test_ts)
    pred_test = denormalize_data(pred_test, dmin, dmax)
    return pred_test

def test_cnn_summary_stats():
    pred_summary_stats = verify_model(model_cnn)
    
    # CNN model should map 100 dim data to 2 dimensions for the test to pass
    assert len(pred_summary_stats[0]) == 2, "CNN Summary Statistics test failed"


# In[ ]:




