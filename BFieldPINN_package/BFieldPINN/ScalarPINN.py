import os
import math
import pickle as pkl
import tensorflow as tf
from BFieldPINN.NN_callbacks import register_x_sin2x_func

class ScalarPINN(tf.keras.Model):
    def __init__(self, norm_dict, x_u, y_u, z_u, validation_data, validation_labels, u_labels, layers_in, activ, snake_a=5., lambda_=0.5,
                 reg=0.0, N_f=50000, tracking_data=None, track_stride=10, initializer=None,
                 colloc_seed=None):
        super(ScalarPINN, self).__init__()

        # rescaling definitions
        self.norm_dict = norm_dict
        # training data
        self.x_u = tf.cast(x_u, dtype = "float32")
        self.y_u = tf.cast(y_u, dtype = "float32")
        self.z_u = tf.cast(z_u, dtype = "float32")
        self.u_labels = tf.cast(u_labels, dtype = "float32")
        # validation data
        self.validation_data = validation_data
        self.x_v = tf.reshape(validation_data[:, 0], (len(validation_data), 1))
        self.y_v = tf.reshape(validation_data[:, 1], (len(validation_data), 1))
        self.z_v = tf.reshape(validation_data[:, 2], (len(validation_data), 1))
        self.validation_labels = validation_labels

        # if we want to track progress on some data
        self.tracking_data = tracking_data
        self.track_stride = track_stride
        if not tracking_data is None:
            self.pred_track = {}
            self.run_track = True
        else:
            self.run_track = False

        self.epoch = 0

        # collocation region definition
        self.L = (z_u.max() - z_u.min())/2.
        self.R = tf.reduce_max((x_u**2 + y_u**2)**(1/2)).numpy()
        self.X0 = (x_u.max() + x_u.min()) / 2.
        self.Y0 = (y_u.max() + y_u.min()) / 2.
        self.Z0 = (z_u.max() + z_u.min()) / 2.
        self.N_f = N_f

        # activation function, hyperparameter setup
        self.activ = activ
        self.snake_a = snake_a
        self.activ_name = None
        if self.activ=='tanh':
            activation_func = tf.keras.activations.tanh
        elif self.activ=='gelu':
            activation_func = tf.keras.activations.gelu
        elif self.activ=='x_sin2x':
            # register the correct function
            self.activ_name = register_x_sin2x_func(a=self.snake_a)
            activation_func = tf.keras.utils.get_custom_objects()[self.activ_name]
        self.lambda_ = tf.Variable(0.0, dtype='float32', trainable=False)
        self.lambda_init = lambda_
        # scale the B during pretraining to avoid early LR reduction
        self.epsilon_ = tf.Variable(100.0, dtype='float32', trainable=False)
        self.reg = reg
        # weight initialization
        if initializer is None:
            self.initializer = ['zeros', 'zeros']
        else:
            self.initializer = initializer

        self.colloc_seed = colloc_seed

        # set up network
        self.layers_in = layers_in
        self.PINN_layers = []
        for i in range(1, len(layers_in)-1):
            if self.reg < 1e-6:
                kernel_reg = None
            else:
                kernel_reg = tf.keras.regularizers.L1(self.reg)
            self.PINN_layers.append(tf.keras.layers.Dense(layers_in[i], activation = activation_func, kernel_regularizer = kernel_reg, trainable = True,
                                                          kernel_initializer=initializer[0], bias_initializer=initializer[1]))
        self.PINN_layers.append(tf.keras.layers.Dense(layers_in[-1], activation = None, trainable = True))
        self(tf.concat([self.x_u, self.y_u, self.z_u], axis = 1))

    def compile(self, optimizer):
        super(ScalarPINN, self).compile()
        self.optimizer = optimizer

    @tf.function
    def update_lambda(self, lambda_):
        self.lambda_.assign(lambda_)

    @tf.function
    def update_epsilon(self, epsilon_):
        self.epsilon_.assign(epsilon_)

    @tf.function
    def call(self, inputs):
        y = inputs
        for i in range(len(self.PINN_layers)):
            y = self.PINN_layers[i](y)
        return y

    @tf.function
    def train_step(self, data):
        # epoch increment so we can check when to add to tracking data
        self.epoch += 1
        # Side length of the cube
        L = self.L
        R = self.R

        # FIXME! This should be passed in as a generator function that returns x_f, y_f, z_f so that different geometries can be used.
        # generate collocation points -- cylindrical
        self.z_f = tf.random.uniform(minval=-L, maxval=L, shape=(self.N_f, 1), seed=self.colloc_seed) + self.Z0
        # R, theta --> x, y
        rs = R * tf.math.sqrt(tf.random.uniform(minval=0, maxval=1, shape=(self.N_f, 1), seed=self.colloc_seed))
        ths = tf.random.uniform(minval=0, maxval=2*math.pi, shape=(self.N_f, 1), seed=self.colloc_seed)
        self.x_f = rs * tf.math.cos(ths) + self.X0
        self.y_f = rs * tf.math.sin(ths) + self.Y0

        # gradients needed for: B = -grad(phi), loss(divB)
        with tf.GradientTape(persistent = True) as g:
            g.watch([self.x_f, self.y_f, self.z_f, self.x_u, self.y_u, self.z_u, self.x_v, self.y_v, self.z_v])

            # calculate B on training data
            inputs_train = tf.concat([self.x_u, self.y_u, self.z_u], axis = 1)
            phi_train = self(inputs_train)
            B_x_train = -g.gradient(phi_train[:, 0], self.x_u) * (2./self.norm_dict['X']['range'])
            B_y_train = -g.gradient(phi_train[:, 0], self.y_u) * (2./self.norm_dict['Y']['range'])
            B_z_train = -g.gradient(phi_train[:, 0], self.z_u) * (2./self.norm_dict['Z']['range'])
            B_train = tf.concat([B_x_train, B_y_train, B_z_train], axis=1)

            # calculate B and div(B) on collocation points
            inputs_f = tf.concat([self.x_f, self.y_f, self.z_f], axis = 1)
            # B
            phi_f = self(inputs_f)
            B_x_f = -g.gradient(phi_f[:, 0], self.x_f) * (2./self.norm_dict['X']['range'])
            B_y_f = -g.gradient(phi_f[:, 0], self.y_f) * (2./self.norm_dict['Y']['range'])
            B_z_f = -g.gradient(phi_f[:, 0], self.z_f) * (2./self.norm_dict['Z']['range'])
            # div(B)
            dBx_dx_f = g.gradient(B_x_f, self.x_f) * (2./self.norm_dict['X']['range'])
            dBy_dy_f = g.gradient(B_y_f, self.y_f) * (2./self.norm_dict['Y']['range'])
            dBz_dz_f = g.gradient(B_z_f, self.z_f) * (2./self.norm_dict['Z']['range'])

            # calculate loss terms
            # B loss
            loss_B = tf.reduce_mean(tf.square(B_train - self.u_labels))
            # div
            div = dBx_dx_f + dBy_dy_f + dBz_dz_f
            loss_div = tf.reduce_mean(tf.square(div))
            # curl is zero by construction
            loss_curl = 0.
            # total loss, using current values of hyperparameters
            lambda_ = self.lambda_
            epsilon_ = self.epsilon_
            loss = epsilon_*(loss_B + lambda_*loss_div)

            # calculate B, loss for validation data
            inputs_v = tf.concat([self.x_v, self.y_v, self.z_v], axis = 1)
            phi_val = self(inputs_v)
            B_x_val = -g.gradient(phi_val[:, 0], self.x_v) * (2./self.norm_dict['X']['range'])
            B_y_val = -g.gradient(phi_val[:, 0], self.y_v) * (2./self.norm_dict['Y']['range'])
            B_z_val = -g.gradient(phi_val[:, 0], self.z_v) * (2./self.norm_dict['Z']['range'])
            B_val = tf.concat([B_x_val, B_y_val, B_z_val], axis=1)
            loss_val = tf.reduce_mean(tf.square(B_val - self.validation_labels))
        # backpropagation
        gradients_1 = g.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_1, self.trainable_variables))

        return {"Loss": loss, "Loss_curl": loss_curl, "Loss_div": loss_div,
        "Loss_B": loss_B, "Loss_val": loss_val, 'lambda_': lambda_, 'epsilon_': epsilon_}

    @tf.function
    def get_B(self, x, y, z):
        with tf.GradientTape(persistent = True) as g:
            g.watch([x, y, z])
            inputs_ = tf.concat([x, y, z], axis=1)
            pred = self(inputs_)
            Bx = -g.gradient(pred[:, 0], x) * (2./self.norm_dict['X']['range'])
            By = -g.gradient(pred[:, 0], y) * (2./self.norm_dict['Y']['range'])
            Bz = -g.gradient(pred[:, 0], z) * (2./self.norm_dict['Z']['range'])
            B_pred = tf.concat([Bx, By, Bz], axis=1)
        return B_pred

    @tf.function
    def get_B_and_Jacobian(self, x, y, z):
        # for calcualting exact derivatives in the model
        with tf.GradientTape(persistent = True) as g:
            g.watch([x, y, z])
            # field
            inputs_ = tf.concat([x, y, z], axis=1)
            pred = self(inputs_)
            Bx = -g.gradient(pred[:, 0], x) * (2./self.norm_dict['X']['range'])
            By = -g.gradient(pred[:, 0], y) * (2./self.norm_dict['Y']['range'])
            Bz = -g.gradient(pred[:, 0], z) * (2./self.norm_dict['Z']['range'])
            B_pred = tf.concat([Bx, By, Bz], axis=1)
            # jacobian
            # row1
            dBx_dx = g.gradient(Bx, x) * (2./self.norm_dict['X']['range'])
            dBx_dy = g.gradient(Bx, y) * (2./self.norm_dict['Y']['range'])
            dBx_dz = g.gradient(Bx, z) * (2./self.norm_dict['Z']['range'])
            jac_Bx = tf.concat([dBx_dx, dBx_dy, dBx_dz], axis=1)
            # row2
            dBy_dx = g.gradient(By, x) * (2./self.norm_dict['X']['range'])
            dBy_dy = g.gradient(By, y) * (2./self.norm_dict['Y']['range'])
            dBy_dz = g.gradient(By, z) * (2./self.norm_dict['Z']['range'])
            jac_By = tf.concat([dBy_dx, dBy_dy, dBy_dz], axis=1)
            # row3
            dBz_dx = g.gradient(Bz, x) * (2./self.norm_dict['X']['range'])
            dBz_dy = g.gradient(Bz, y) * (2./self.norm_dict['Y']['range'])
            dBz_dz = g.gradient(Bz, z) * (2./self.norm_dict['Z']['range'])
            jac_Bz = tf.concat([dBz_dx, dBz_dy, dBz_dz], axis=1)
            # concatenate
            jac = tf.stack([jac_Bx, jac_By, jac_Bz], axis=1) # N_data x 3 (dB_i) x 3 (dx_j)
            return B_pred, jac

    def save_model(self, model_fname):
        if not os.path.exists(model_fname):
            os.makedirs(model_fname)
        # construct init dict
        config = {
            'norm_dict': self.norm_dict,
            'x_u': self.x_u.numpy(),
            'y_u': self.y_u.numpy(),
            'z_u': self.z_u.numpy(),
            'validation_data': self.validation_data,
            'validation_labels': self.validation_labels,
            'u_labels': self.u_labels,
            'layers_in': self.layers_in,
            'activ': self.activ,
            'snake_a': self.snake_a,
            'lambda_': self.lambda_,
            'reg': self.reg,
            'N_f': self.N_f,
            'tracking_data': self.tracking_data,
            'track_stride': self.track_stride,
            'initializer': self.initializer,
            'colloc_seed': self.colloc_seed,
        }
        # configs
        pkl.dump(config, open(model_fname+'/init_config.pkl', 'wb'))
        # weights
        self.save_weights(model_fname+'/trained_weights')

    @classmethod
    def load_model(cls, model_fname):
        config = pkl.load(open(model_fname+'/init_config.pkl', 'rb'))
        inst = cls(**config)
        inst.load_weights(model_fname+'/trained_weights').expect_partial()
        return inst
