import numpy as np
import tensorflow as tf
import edward as ed
import edward.models as edm


# node names in lexicographic (alphabetic) order
nd_names_lex_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

# node names in topological (chronological) order
nd_names_topo_ord = ['Cloudy', 'Rain', 'Sprinkler', 'WetGrass']

with tf.name_scope('model'):
    Cloudy = tf.placeholder(tf.int32, shape=[sam_size],
        name="Cloudy")

    alpha_Rain = np.array([[ 1.,  1.],
       [ 1.,  1.]])
    probs_Rain = edm.Dirichlet(
        alpha_Rain.astype(np.float32), name='probs_Rain')
    p_Rain = tf.stack([
        probs_Rain[Cloudy[j], :]
        for j in range(sam_size)])
    Rain = edm.Categorical(
        probs=p_Rain, name='Rain')

    alpha_Sprinkler = np.array([[ 1.,  1.],
       [ 1.,  1.]])
    probs_Sprinkler = edm.Dirichlet(
        alpha_Sprinkler.astype(np.float32), name='probs_Sprinkler')
    p_Sprinkler = tf.stack([
        probs_Sprinkler[Cloudy[j], :]
        for j in range(sam_size)])
    Sprinkler = edm.Categorical(
        probs=p_Sprinkler, name='Sprinkler')

    arr_WetGrass = np.array([[[ 0.99,  0.01],
        [ 0.01,  0.99]],

       [[ 0.01,  0.99],
        [ 0.01,  0.99]]])
    ten_WetGrass = tf.convert_to_tensor(arr_WetGrass, dtype=tf.float32)
    p_WetGrass = tf.stack([
        ten_WetGrass[Sprinkler[j], Rain[j], :]
        for j in range(sam_size)])
    WetGrass = edm.Categorical(
        probs=p_WetGrass, name='WetGrass')

with tf.name_scope('posterior'):
    # Cloudy = placeholder

    probs_Rain_q = edm.Dirichlet(tf.nn.softplus(
        tf.get_variable('pos_Rain_q', shape=(2, 2))),
        name='probs_Rain_q')

    probs_Sprinkler_q = edm.Dirichlet(tf.nn.softplus(
        tf.get_variable('pos_Sprinkler_q', shape=(2, 2))),
        name='probs_Sprinkler_q')

    WetGrass_ph = tf.placeholder(tf.int32, shape=[sam_size],
        name="WetGrass_ph")

