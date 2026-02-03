import tensorflow as tf

# -------- Force CPU only --------
tf.config.set_visible_devices([], 'GPU')


class DSFANet(tf.keras.Model):
    def __init__(self, num_samples, output_num=6, hidden_num=128, layers=2, reg=1e-4):
        super().__init__()

        self.num = num_samples
        self.output_num = output_num
        self.hidden_num = hidden_num
        self.layers_num = layers
        self.reg = reg

        self.activation = tf.nn.softsign
        self.init = tf.keras.initializers.HeNormal()

        # Shared dense stacks for X and Y
        self.hidden_layers = [
            tf.keras.layers.Dense(
                hidden_num,
                activation=self.activation,
                kernel_initializer=self.init,
                use_bias=True
            )
            for _ in range(self.layers_num)
        ]

        self.out_layer = tf.keras.layers.Dense(
            output_num,
            activation=self.activation,
            kernel_initializer=self.init,
            use_bias=True
        )

    def DSFA(self, X, Y):
        # Center the data
        X_hat = X - tf.reduce_mean(X, axis=0)
        Y_hat = Y - tf.reduce_mean(Y, axis=0)

        differ = X_hat - Y_hat

        # A matrix
        A = tf.matmul(differ, differ, transpose_a=True)
        A = A / tf.cast(self.num, tf.float32)

        # Covariance matrices
        Sigma_XX = tf.matmul(X_hat, X_hat, transpose_a=True)
        Sigma_XX = Sigma_XX / tf.cast(self.num, tf.float32)
        Sigma_XX += self.reg * tf.eye(self.output_num)

        Sigma_YY = tf.matmul(Y_hat, Y_hat, transpose_a=True)
        Sigma_YY = Sigma_YY / tf.cast(self.num, tf.float32)
        Sigma_YY += self.reg * tf.eye(self.output_num)

        B = (Sigma_XX + Sigma_YY) / 2.0

        # Eigen decomposition (symmetric matrix)
        D_B, V_B = tf.linalg.eigh(B)

        # Numerical stability
        mask = D_B > 1e-12
        D_B = tf.boolean_mask(D_B, mask)
        V_B = tf.boolean_mask(V_B, mask, axis=1)

        B_inv = V_B @ tf.linalg.diag(tf.math.reciprocal(D_B)) @ tf.transpose(V_B)

        Sigma = B_inv @ A
        loss = tf.linalg.trace(Sigma @ Sigma)

        return loss

    def call(self, inputs, training=False):
        X, Y = inputs

        for layer in self.hidden_layers:
            X = layer(X)
            Y = layer(Y)

        X_out = self.out_layer(X)
        Y_out = self.out_layer(Y)

        loss = self.DSFA(X_out, Y_out)
        return loss


