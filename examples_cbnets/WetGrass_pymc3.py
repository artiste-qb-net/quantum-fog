import numpy as np
import pymc3 as pm3
import theano
import theano.tensor as tt


# node names in lexicographic (alphabetic) order
nd_names_lex_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

# node names in topological (chronological) order
nd_names_topo_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

data_Cloudy = [1]

data_Rain = None

data_Sprinkler = None

data_WetGrass = [0, 1]


mod = pm3.Model()
with mod:
    p_Cloudy = np.array([ 0.5,  0.5])
    Cloudy = pm3.Categorical(
        'Cloudy', p=p_Cloudy, observed=data_Cloudy)

    arr_Rain = np.array([[ 0.4,  0.6],
       [ 0.5,  0.5]])
    p_Rain = theano.shared(arr_Rain)[
        Cloudy, :]
    Rain = pm3.Categorical(
        'Rain', p=p_Rain, observed=data_Rain)

    arr_Sprinkler = np.array([[ 0.2,  0.8],
       [ 0.7,  0.3]])
    p_Sprinkler = theano.shared(arr_Sprinkler)[
        Cloudy, :]
    Sprinkler = pm3.Categorical(
        'Sprinkler', p=p_Sprinkler, observed=data_Sprinkler)

    arr_WetGrass = np.array([[[ 0.99,  0.01],
        [ 0.01,  0.99]],

       [[ 0.01,  0.99],
        [ 0.01,  0.99]]])
    p_WetGrass = theano.shared(arr_WetGrass)[
        Sprinkler, Rain, :]
    WetGrass = pm3.Categorical(
        'WetGrass', p=p_WetGrass, observed=data_WetGrass)

