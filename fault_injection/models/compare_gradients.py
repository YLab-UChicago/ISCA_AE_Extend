import tensorflow as tf

class CompareGradients(tf.keras.Model):
    def __init__(self):
        super(CompareGradients, self).__init__()

    def call(self, golden_gradients, manual_gradients):
        cmp_range = len(manual_gradients)

        diffs = []
        maxes = []
        for i in range(cmp_range):
            g_grad = golden_gradients[i]
            m_grad = manual_gradients[i]

            #print("The ith gradient: golden shape {}, manual shape {}".format(g_grad.get_shape(), m_grad.get_shape()))

            num_diff = tf.reduce_sum(tf.cast(tf.greater(tf.abs(tf.subtract(g_grad, m_grad)), 0.0001), tf.int32))
            diffs.append(num_diff)

            max_error = tf.reduce_max(tf.abs(m_grad))
            #max_error = tf.reduce_max(tf.abs(tf.subtract(g_grad, m_grad)))
            maxes.append(max_error)

        return diffs, maxes
