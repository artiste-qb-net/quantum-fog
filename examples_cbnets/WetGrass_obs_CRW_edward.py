import numpy as np
import tensorflow as tf
import edward as ed
import edward.models as edm


# node names in lexicographic (alphabetic) order
nd_names_lex_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

# node names in topological (chronological) order
nd_names_topo_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

# dominant, most common state
def domi(rv):
    return tf.argmax(tf.bincount(rv))

with tf.name_scope('model'):
    Cloudy = tf.placeholder(tf.int32, shape=[sam_size],
        name="Cloudy")

    arr_Rain = np.array([[ 0.4,  0.6],
       [ 0.5,  0.5]])
    ten_Rain = tf.convert_to_tensor(arr_Rain, dtype=tf.float32)
    p_Rain = tf.stack([
        ten_Rain[Cloudy[j], :]
        for j in range(sam_size)])
    Rain = edm.Categorical(
        probs=p_Rain, name='Rain')

    alpha_Sprinkler = np.array([[ 1.,  1.],
       [ 1.,  1.]])
    probs_Sprinkler = edm.Dirichlet(
        alpha_Sprinkler.astype(np.float32), name='probs_Sprinkler')
    p_Sprinkler = probs_Sprinkler[domi(Cloudy), :]
    Sprinkler = edm.Categorical(
        probs=p_Sprinkler, name='Sprinkler')

    arr_WetGrass = np.array([[[ 0.99,  0.01],
        [ 0.01,  0.99]],

       [[ 0.01,  0.99],
        [ 0.01,  0.99]]])
    ten_WetGrass = tf.convert_to_tensor(arr_WetGrass, dtype=tf.float32)
    p_WetGrass = tf.stack([
        ten_WetGrass[Sprinkler, Rain[j], :]
        for j in range(sam_size)])
    WetGrass = edm.Categorical(
        probs=p_WetGrass, name='WetGrass')

with tf.name_scope('posterior'):
    # Cloudy = placeholder

    Rain_ph = tf.placeholder(tf.int32, shape=[sam_size],
        name="Rain_ph")

    probs_Sprinkler_q = edm.Dirichlet(
        tf.nn.softplus(tf.get_variable('pos_Sprinkler_q', shape=(2, 2))),
        name='probs_Sprinkler_q')

    WetGrass_ph = tf.placeholder(tf.int32, shape=[sam_size],
        name="WetGrass_ph")

