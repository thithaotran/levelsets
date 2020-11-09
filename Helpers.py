import tensorflow as tf
import os
import warnings

def setup_logs(experiment_name):
    '''
    sets up log dir for an experiment
    '''

    experiment_index_start = 0
    if not (experiment_name in os.listdir('.')):
        os.mkdir(experiment_name)
    else:
        path_experiment = os.path.join('.', experiment_name)
        warnings.warn('Path {} already exists!'.format(path_experiment))
        if len(os.listdir(path_experiment)):
            list_npz = [f for f in os.listdir(path_experiment) if f.endswith('npz')]
            if len(list_npz):
                temp = max(list_npz)
                experiment_index_start = 1 + int((temp.split('_')[-1]).split('.')[0])
    return experiment_index_start

def recover_individual_grads(concatenated_grads, model):
    '''
    recover individual grads/grad-like slices from concatenated predictor and corrector step vectors
    '''
    print('tracing recover_individual_grads') # will run only the first time (function is added to the graph)
    list_grads = []
    index_extract = 0
    for var in model.trainable_variables:
        slice_for_var = tf.reshape(concatenated_grads[index_extract:index_extract+tf.math.reduce_prod(var.shape)], var.shape)

        list_grads.append(slice_for_var)

        index_extract += tf.math.reduce_prod(var.shape)
    return list_grads


def get_angle(t1, t2, unitvecs=False):
    '''
    Given two 1-D tensors, returns the arccosine of the angle between them. numpy equivalents are commented out.
    '''
    print('tracing get_angle')
    if unitvecs: # if given two vectors have already been standarized
        t1_norm = tf.constant(1.0)
        t2_norm = tf.constant(1.0)
    else:
        t1_norm = tf.linalg.norm(t1) #np.sqrt(np.sum(np.square(t1)))
        t2_norm =tf.linalg.norm(t2) #np.sqrt(np.sum(np.square(t2)))

    t1_dot_t2 = tf.reduce_sum(tf.multiply(t1, t2)) # np.sum(t1 * t2)

    angle =  tf.math.acos(tf.clip_by_value(t1_dot_t2/(t1_norm*t2_norm),
                                           clip_value_min = -1.0,
                                           clip_value_max = 1.0)) # np.arccos(np.clip(t1_dot_t2/(t1_norm*t2_norm), -1.0, 1.0))

    return angle

def get_rejection(vec_a, vec_b, unit_b = False):
    print('tracing get_rejection')
    if not unit_b:
        unit_vec_b = vec_b/tf.linalg.norm(vec_b)
    else:
        unit_vec_b = vec_b

    rejection = vec_a - tf.reduce_sum(vec_a*unit_vec_b)*unit_vec_b

    return rejection


class Log:
    # Lists that keep logs of different model statistics
    def __init__(self):
        self.SEED = None
        # epoch indices
        self.index_epoch = []
        self.index_epoch_pred = []
        self.index_epoch_corr = []

        # losses and metrics
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        # stuff to track for more plots
        self.loss_change = []
        self.param_norms = []
        self.angle_bw_gradients = []

        # learning rates
        self.lr_pred = []
        self.lr_corr = []

        # weights
        self.weights_phase1 = []
        self.weights_phase2 = []

if __name__ == "__main__":
    pass