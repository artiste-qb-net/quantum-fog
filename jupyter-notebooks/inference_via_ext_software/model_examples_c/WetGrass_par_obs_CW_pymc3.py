import numpy as np
import theano as th
import theano.tensor as tt
import pymc3 as pm3


# node names in lexicographic (alphabetic) order
nd_names_lex_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

# node names in topological (chronological) order
nd_names_topo_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

# data_Cloudy = 

# data_WetGrass = 


mod = pm3.Model()
with mod:
    arr_Cloudy = np.array([ 0.5,  0.5])
    ten_Cloudy = th.shared(arr_Cloudy)
    p_Cloudy = ten_Cloudy[:]
    Cloudy = pm3.Categorical(
        'Cloudy', p=p_Cloudy, observed=data_Cloudy, shape=sam_size)

    alpha_Rain = np.array([[ 1.,  1.],
       [ 1.,  1.]])
    probs_Rain = pm3.Dirichlet(
        'probs_Rain', a=alpha_Rain, shape=(2, 2))
    p_Rain = tt.stack([
        probs_Rain[Cloudy[j], :]
        for j in range(sam_size)])
    Rain = pm3.Categorical(
        'Rain', p=p_Rain, shape=sam_size)

    alpha_Sprinkler = np.array([[ 1.,  1.],
       [ 1.,  1.]])
    probs_Sprinkler = pm3.Dirichlet(
        'probs_Sprinkler', a=alpha_Sprinkler, shape=(2, 2))
    p_Sprinkler = tt.stack([
        probs_Sprinkler[Cloudy[j], :]
        for j in range(sam_size)])
    Sprinkler = pm3.Categorical(
        'Sprinkler', p=p_Sprinkler, shape=sam_size)

    arr_WetGrass = np.array([[[ 0.99,  0.01],
        [ 0.01,  0.99]],

       [[ 0.01,  0.99],
        [ 0.01,  0.99]]])
    ten_WetGrass = th.shared(arr_WetGrass)
    p_WetGrass = tt.stack([
        ten_WetGrass[Sprinkler[j], Rain[j], :]
        for j in range(sam_size)])
    WetGrass = pm3.Categorical(
        'WetGrass', p=p_WetGrass, observed=data_WetGrass, shape=sam_size)

