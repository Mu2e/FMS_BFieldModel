# interfaces for ScalarPINN and StandardPINN are the same, so we can reuse the
# custom callbacks. Also includes snake function activation registration
import numpy as np
import tensorflow as tf

# create callback list to pass in at fit
def get_callback_list(PINN_inst, NN_dict):
    callbacks = []
    # temper / pretrain lambda
    temperLambdaCallback = TemperLambda(PINN_inst,
        N_pretrain=NN_dict['lambda_pretrain'], N_wait=NN_dict['lambda_N_wait'],
        mult_factor=NN_dict['lambda_mult_factor'], add_factor=NN_dict['lambda_add_factor'],
        start_temper_epoch=NN_dict['lambda_start_temper'], max_lambda=NN_dict['lambda_max'])
    callbacks.append(temperLambdaCallback)
    # track
    if NN_dict['track']:
        predictionTrackCallback = PredictionTrack(PINN_inst)
        callbacks.append(predictionTrackCallback)
    # LR reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=NN_dict['LR_monitor'], factor=NN_dict['LR_factor'],
        patience=NN_dict['LR_patience'], min_lr=NN_dict['LR_min'])
    callbacks.append(reduce_lr)
    # early stop
    trainingStopCallback = tf.keras.callbacks.EarlyStopping(
        monitor=NN_dict['Stop_monitor'], patience=NN_dict['Stop_patience'],
        min_delta=NN_dict['Stop_min_delta'])
    callbacks.append(trainingStopCallback)
    return callbacks

# evaluating model and jacobian
# should work for either type of PINN.
def model_predict(PINN_inst, x, y, z, N_i=100000):
    N_chunk = len(x) // N_i + 1
    pred_list = []
    for i in range(N_chunk):
        i0 = i*N_i
        i1 = (i+1)*N_i
        N_ = len(x[i0:i1])
        x_ = tf.cast(x[i0:i1].reshape(N_, 1), dtype=tf.float32)
        y_ = tf.cast(y[i0:i1].reshape(N_, 1), dtype=tf.float32)
        z_ = tf.cast(z[i0:i1].reshape(N_, 1), dtype=tf.float32)
        pred_ = PINN_inst.get_B(x_, y_, z_).numpy()
        pred_list.append(pred_)
    pred_test = np.concatenate(pred_list, axis=0)
    return pred_test, None

def model_predict_with_jacobian(PINN_inst, x, y, z, N_i=100000):
    N_chunk = len(x) // N_i + 1
    pred_list = []
    jac_list = []
    for i in range(N_chunk):
        i0 = i*N_i
        i1 = (i+1)*N_i
        N_ = len(x[i0:i1])
        x_ = tf.cast(x[i0:i1].reshape(N_, 1), dtype=tf.float32)
        y_ = tf.cast(y[i0:i1].reshape(N_, 1), dtype=tf.float32)
        z_ = tf.cast(z[i0:i1].reshape(N_, 1), dtype=tf.float32)
        pred_, jac_ = PINN_inst.get_B_and_Jacobian(x_, y_, z_)
        pred_list.append(pred_.numpy())
        jac_list.append(jac_.numpy())
    pred_test = np.concatenate(pred_list, axis=0)
    jac_test = np.concatenate(jac_list, axis=0)
    return pred_test, jac_test

# snake function register
def register_x_sin2x_func(a=1):
    K = tf.keras.backend
    def x_sin2x(x):
        return x + K.square(K.sin(a*x))
    activ_name = f'x_sin2x_a{a:0.1f}'
    tf.keras.utils.get_custom_objects().update({activ_name: tf.keras.layers.Activation(x_sin2x)})
    return activ_name

class PredictionTrack(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        if self.model.run_track:
            if (epoch % self.model.track_stride == 0):
                self.model.pred_track[epoch] = self.model.get_B(self.model.tracking_data[:,0:1], self.model.tracking_data[:,1:2], self.model.tracking_data[:,2:3]).numpy()

# Lambda adjustments include: tempering, pretraining.
class TemperLambda(tf.keras.callbacks.Callback):
    def __init__(self, model, N_pretrain=500, N_wait=20000, mult_factor=2.0, add_factor=0.0, start_temper_epoch=1, max_lambda=None):
        self.model = model
        self.lambda_init = model.lambda_init
        self.N_pretrain = N_pretrain
        self.N_wait = N_wait
        self.mult_factor = np.float32(mult_factor)
        self.add_factor = np.float32(add_factor)
        self.start_temper_epoch = start_temper_epoch
        if not max_lambda is None:
            self.max_lambda = np.float32(max_lambda)
        else:
            self.max_lambda = max_lambda

    def on_epoch_end(self, epoch, logs={}):
        # first two conditionals handle pretraining. everything after is tempering
        if epoch < self.N_pretrain:
            pass
        elif epoch == self.N_pretrain:
            self.model.update_lambda(self.lambda_init)
            # update epsilon from 100 to 1 to avoid early LR reduction due to lambda adjustment
            self.model.update_epsilon(1.0)
        else:
            if (epoch % self.N_wait == 0) & (epoch > self.start_temper_epoch):
                new_lambda_ = self.model.lambda_*self.mult_factor+self.add_factor
                if not self.max_lambda is None:
                    if (new_lambda_) > self.max_lambda:
                        val = self.max_lambda
                    else:
                        val = new_lambda_
                else:
                    val = new_lambda_
                self.model.update_lambda(val)
