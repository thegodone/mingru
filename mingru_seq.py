import tensorflow as tf

def g(x):
    return tf.where(x >= 0, x + 0.5, tf.sigmoid(x))

class minGRU(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, expansion_factor=1.0):
        super(minGRU, self).__init__()
        self.units = units
        self.input_dim = input_dim
        dim_inner = int(self.units * expansion_factor)
        
        # Initialize weights for the combined hidden state and gate calculations
        self.W = self.add_weight(shape=(input_dim, dim_inner * 2), initializer='glorot_uniform', trainable=True, name='W')
        if expansion_factor != 1:
            self.W_out = self.add_weight(shape=(dim_inner, units), initializer='glorot_uniform', trainable=True, name='W_out')
        else:
            self.W_out = None  # Use identity mapping if no expansion

    def call(self, x, h_prev=None):
        # Apply linear transformation and split into hidden state and gate components
        z_h = tf.matmul(x, self.W)
        z, h_tilde = tf.split(z_h, num_or_size_splits=2, axis=1)

        # Apply custom non-linearities
        h_tilde = g(h_tilde)
        z = tf.sigmoid(z)

        # Perform element-wise interpolation between previous hidden state and new candidate
        if h_prev is not None:
            h = (1 - z) * h_prev + z * h_tilde
        else:
            h = z * h_tilde  # If no previous state, use the new state scaled by the update gate

        # Apply output transformation if necessary
        if self.W_out is not None:
            h = tf.matmul(h, self.W_out)

        return h

# Example usage
units = 128
input_dim = 64
layer = minGRU(units=units, input_dim=input_dim)

# Test the layer
x = tf.random.normal((1, input_dim))  # Example input tensor
h_prev = tf.random.normal((1, units))  # Previous hidden state

h_next = layer(x, h_prev)
print(h_next.shape)  # Expected output shape: (1, units)
