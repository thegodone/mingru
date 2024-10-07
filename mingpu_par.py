# code is not working
import tensorflow as tf

def g(x):
    """ Custom activation function used in the model. """
    return tf.where(x >= 0, x + 0.5, tf.sigmoid(x))

def log_g(x):
    """ Log-space version of the custom activation function. """
    return tf.where(x >= 0, tf.math.log(tf.nn.relu(x) + 0.5), -tf.math.softplus(-x))

def heinsen_associative_scan_log(log_coeffs, log_values):
    """ Parallel scan in log-space as per Heinsen's approach. """
    a_star = tf.cumsum(log_coeffs, axis=1)
    log_h0_plus_b_star = tf.experimental.numpy.logcumsumexp(log_values - a_star, axis=1)
    log_h = a_star + log_h0_plus_b_star
    return tf.exp(log_h)

class minGRU(tf.keras.layers.Layer):
    def __init__(self, dim, expansion_factor=1.):
        super(minGRU, self).__init__()
        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = tf.keras.layers.Dense(dim_inner * 2, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim, use_bias=False) if expansion_factor != 1. else tf.keras.layers.Lambda(lambda x: x)

    def call(self, x, prev_hidden=None):
        hidden_gate = self.to_hidden_and_gate(x)
        hidden, gate = tf.split(hidden_gate, 2, axis=-1)
    
        log_coeffs = -tf.nn.softplus(gate)
        log_z = -tf.nn.softplus(-gate)
        log_tilde_h = log_g(hidden)
        log_values = log_z + log_tilde_h
    
        if prev_hidden is not None:
            prev_hidden_log = log_g(prev_hidden)
            prev_hidden_log = tf.expand_dims(prev_hidden_log, axis=1)  # Shape: (batch_size, 1, units)
            prev_hidden_log = tf.tile(prev_hidden_log, [1, tf.shape(x)[1], 1])  # Repeat to match sequence length
            log_values = tf.concat([prev_hidden_log, log_values], axis=1)
            log_coeffs = tf.pad(log_coeffs, [[0, 0], [1, 0], [0, 0]])
    
        out = heinsen_associative_scan_log(log_coeffs, log_values)
        out = out[:, -tf.shape(x)[1]:]  # Only keep the last seq_len outputs to match input sequence length
    
        return self.to_out(out)


# Example usage
dim = 128
expansion_factor = 1.
batch_size = 32
seq_len = 10
input_dim = 64

x = tf.random.normal((batch_size, seq_len, input_dim))
prev_hidden = tf.random.normal((batch_size, dim))

layer = minGRU(dim, expansion_factor)
output = layer(x, prev_hidden)
print(output.shape)  # Should match (batch_size, seq_len, dim)
