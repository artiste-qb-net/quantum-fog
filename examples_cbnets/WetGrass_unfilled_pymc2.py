import numpy as np
import pymc as pm2


# node names in lexicographic (alphabetic) order
nd_names_lex_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

# node names in topological (chronological) order
nd_names_topo_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

# did_obs_Cloudy = False
# data_Cloudy = None

# did_obs_Rain = False
# data_Rain = None

# did_obs_Sprinkler = False
# data_Sprinkler = None

# did_obs_WetGrass = False
# data_WetGrass = None


p_Cloudy = np.array([ 0.5,  0.5])
Cloudy = pm2.Categorical(
    'Cloudy', p=p_Cloudy,
    value=data_Cloudy, observed=did_obs_Cloudy)


@pm2.deterministic
def p_Rain(Cloudy1=Cloudy):
    Cloudy1 = Cloudy
    k0 = int(Cloudy1.value), 

    arr = np.array([[ 0.4,  0.6],
       [ 0.5,  0.5]])
    return arr[k0, :]
Rain = pm2.Categorical(
    'Rain', p=p_Rain,
    value=data_Rain, observed=did_obs_Rain)


@pm2.deterministic
def p_Sprinkler(Cloudy1=Cloudy):
    Cloudy1 = Cloudy
    k0 = int(Cloudy1.value), 

    arr = np.array([[ 0.2,  0.8],
       [ 0.7,  0.3]])
    return arr[k0, :]
Sprinkler = pm2.Categorical(
    'Sprinkler', p=p_Sprinkler,
    value=data_Sprinkler, observed=did_obs_Sprinkler)


@pm2.deterministic
def p_WetGrass(Sprinkler1=Sprinkler, Rain1=Rain):
    Sprinkler1 = Sprinkler
    k0 = int(Sprinkler1.value), 

    Rain1 = Rain
    k1 = int(Rain1.value), 

    arr = np.array([[[ 0.99,  0.01],
        [ 0.01,  0.99]],

       [[ 0.01,  0.99],
        [ 0.01,  0.99]]])
    return arr[k0, k1, :]
WetGrass = pm2.Categorical(
    'WetGrass', p=p_WetGrass,
    value=data_WetGrass, observed=did_obs_WetGrass)


