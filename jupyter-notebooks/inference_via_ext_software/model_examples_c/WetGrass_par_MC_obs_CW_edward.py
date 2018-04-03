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

    emp_Rain_q = edm.Empirical(tf.nn.softmax(
        tf.get_variable('var_Rain_q', shape=(sam_size, 2, 2),
        initializer=tf.constant_initializer(0.5))),
        name='emp_Rain_q')
    propo_Rain_q = edm.Normal(loc=emp_Rain_q, scale=0.05)

    emp_Sprinkler_q = edm.Empirical(tf.nn.softmax(
        tf.get_variable('var_Sprinkler_q', shape=(sam_size, 2, 2),
        initializer=tf.constant_initializer(0.5))),
        name='emp_Sprinkler_q')
    propo_Sprinkler_q = edm.Normal(loc=emp_Sprinkler_q, scale=0.05)

    WetGrass_ph = tf.placeholder(tf.int32, shape=[sam_size],
        name="WetGrass_ph")

