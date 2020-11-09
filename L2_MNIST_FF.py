import numpy as np
import os


import tensorflow as tf
from tensorflow import keras
import Configs_L2
import Datasets
import Models_L2
from Helpers import recover_individual_grads, get_rejection, get_angle, setup_logs

# experiment name is the same as the file/script name
# the corresponding Config class is loaded dynamically
exp_name = os.path.basename(__file__).split('.')[0]
cfg = getattr(Configs_L2, "Config_{}".format(exp_name[3:]))() # 3: for L2_ substring
cfg.EXP_INDEX_START = setup_logs(exp_name)

# decorate the imported helper functions with tf.function
recover_individual_grads, get_rejection, get_angle = tf.function(recover_individual_grads), tf.function(get_rejection), tf.function(get_angle)

# import get_datasets and get_model functions dynamically
get_datasets = getattr(Datasets, "get_datasets_{}".format(exp_name[3:]))
get_model = getattr(Models_L2, "get_model_{}".format(exp_name[3:]))

class Log:
    SEED = []
    # losses and metrics
    weights = []

for idx, reg_pen in enumerate(cfg.RANGE_REG_PENALTY, start=1):
        for exp_idx in range(1, cfg.NUM_EXP_TO_RUN+1):
                lg = Log()

                seed = np.random.randint(low=0, high=1000)
                tf.random.set_seed(seed)
                lg.SEED = np.array([seed])

                dataset_train, dataset_test = get_datasets()
                cardinality_train = dataset_train.cardinality().numpy()
                cardinality_test = dataset_test.cardinality().numpy()

                dataset_train = dataset_train.shuffle(buffer_size=cardinality_train, seed=seed)
                dataset_train = dataset_train.take(1000).cache().batch(32)
                dataset_test = dataset_test.cache().batch(min(1000, int(cardinality_test / 2)))


                model = get_model(seed, reg_pen)

                hist = model.fit(dataset_train,
                             validation_data=dataset_test, validation_steps=None,
                             epochs=cfg.NUM_EPOCHS,
                                 verbose=False)

                lg.weights = np.hstack([var.numpy().ravel() for var in model.trainable_variables])

                names_logs_to_save = [att for att in dir(lg) if '__' not in att]
                logs_to_save = [getattr(lg, temp) for temp in names_logs_to_save]
                names_logs_to_save += sorted(list(hist.history.keys()))
                logs_to_save += [hist.history[k] for k in sorted(list(hist.history.keys()))]
                dict_logs_to_save = dict(zip(names_logs_to_save, logs_to_save))
                temp_path = os.path.join(exp_name, 'logs_regpenidx_{}_expidx_{}'.format(idx, exp_idx))
                np.savez(temp_path, **dict_logs_to_save)

                print('Exp {} done!'.format(exp_idx))

