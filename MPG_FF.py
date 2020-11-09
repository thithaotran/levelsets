import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import Configs
import Datasets
import Models
from Helpers import recover_individual_grads, get_rejection, get_angle, setup_logs, Log

# experiment name is the same as the file/script name
# the corresponding Config class is loaded dynamically
exp_name = os.path.basename(__file__).split('.')[0]
cfg = getattr(Configs, "Config_{}".format(exp_name))()
cfg.EXP_INDEX_START = setup_logs(exp_name)

# decorate the imported helper functions with tf.function
recover_individual_grads, get_rejection, get_angle = tf.function(recover_individual_grads), tf.function(get_rejection), tf.function(get_angle)

# import get_datasets and get_model functions dynamically
get_datasets = getattr(Datasets, "get_datasets_{}".format(exp_name))
get_model = getattr(Models, "get_model_{}".format(exp_name))


def get_tape(ds, lss_phs1):
    with tf.GradientTape(persistent=True) as tape:
        train_loss = 0.0

        for x, y in dataset_train:
            train_loss += tf.reduce_sum(tf.square(y - tf.reshape(model(x), shape=[-1])))
        train_loss = train_loss / cardinality_train

        loss_change = tf.square(train_loss - LOSS_PHASE1)

        param_norms = 0
        for p in model.trainable_variables:
            param_norms += tf.reduce_sum(tf.square(p))
    return tape, train_loss, loss_change, param_norms

for exp_idx in range(cfg.EXP_INDEX_START, cfg.EXP_INDEX_START + cfg.NUM_EXP_TO_RUN):

    # a random int in [0, 1000] is generated as a seed for each experiment
    seed = np.random.randint(low=0, high=1000)
    tf.random.set_seed(seed)

    dataset_train, dataset_test = get_datasets()
    cardinality_train = dataset_train.cardinality().numpy()
    cardinality_test = dataset_test.cardinality().numpy()

    dataset_train = dataset_train.shuffle(buffer_size=cardinality_train, seed=seed)
    dataset_train = dataset_train.take(cardinality_train).cache().batch(32)
    dataset_test = dataset_test.cache().batch(min(1000, int(cardinality_test/2)))

    lg = Log()
    #print(dir(lg))
    lg.SEED = np.array([seed])


    # PHASE 1

    model = get_model(seed)

    history = model.fit(dataset_train,
                        validation_data=dataset_test, validation_steps=None,
                        epochs=cfg.NUM_EPOCHS_PHASE1, verbose=0)

    LOSS_PHASE1 = model.evaluate(dataset_train, verbose=False)[0]

    lg = Log()
    lg.weights_phase1 = np.hstack([var.numpy().ravel() for var in model.trainable_variables])

    # PHASE 2

    tape, train_loss, loss_change, param_norms = get_tape(dataset_train, LOSS_PHASE1)

    # regularization gradient (concatenated)
    arrays_grads_param_norms = [tape.gradient(param_norms, p) for p in model.trainable_variables]
    grad_param_norms = tf.concat([tf.reshape(tv, [-1]) for tv in arrays_grads_param_norms], axis=0)

    # training loss gradient (concatenated)
    arrays_grads_loss = [tape.gradient(train_loss, p) for p in model.trainable_variables]
    grad_loss = tf.concat([tf.reshape(tv, [-1]) for tv in arrays_grads_loss], axis=0)

    cfg.ANGLE_PHASE1 = get_angle(grad_param_norms, grad_loss, unitvecs=False).numpy() * 180 / np.pi

    epoch = 0
    epoch_pred = 0
    last = None
    buffer_direction_change_pred = []
    rejection_pred_prev = None
    buffer_direction_change_corr = []
    rejection_corr_prev = None

    # learning rate and optimizer stuff
    lr_pred_min = 1e-5
    lr_pred_max = 0.1
    lr_corr_min = 1e-6
    lr_corr_max = 0.1  # 0.1

    # current rates
    lr_pred_curr = lr_pred_min
    lr_corr_curr = lr_corr_min

    model.optimizer = keras.optimizers.SGD(lr_pred_curr)

    loop_condition = ((not lg.angle_bw_gradients) or (lg.angle_bw_gradients[-1] < 180.0)) and (
                epoch_pred < cfg.NUM_EPOCHS_PHASE2)
    # print(LOSS_PHASE1, train_loss)

    while loop_condition:
        epoch += 1
        lg.index_epoch.append(epoch)

        tape, train_loss, loss_change, param_norms = get_tape(dataset_train, LOSS_PHASE1)
        lg.loss_change.append(loss_change.numpy())

        if cfg.SHOW_PROGRESS:
            print('loss change {}'.format(loss_change.numpy()))

        if loss_change.numpy() <= cfg.THRESHOLD_PHASE2:  #
            if cfg.SHOW_PROGRESS:
                print('@@@ PREDICTION @@')

            epoch_pred += 1
            # regularization gradient (concatenated)
            arrays_grads_param_norms = [tape.gradient(param_norms, p) for p in model.trainable_variables]
            grad_param_norms = tf.concat([tf.reshape(tv, [-1]) for tv in arrays_grads_param_norms],
                                         axis=0)  # np.hstack(arrays_grads_param_norms)

            # training loss gradient (concatenated)
            arrays_grads_loss = [tape.gradient(train_loss, p) for p in model.trainable_variables]
            grad_loss = tf.concat([tf.reshape(tv, [-1]) for tv in arrays_grads_loss], axis=0)

            # the angle b/w the two gradients
            angle = get_angle(grad_param_norms, grad_loss, unitvecs=False).numpy() * 180 / np.pi

            if cfg.SHOW_PROGRESS:
                print('Current angle:', angle)

            # evaluate model and log different stuff, NOTE: THE CURRENT POINT IS ON THE LEVEL SET BEING TRAVERSED!

            lg.angle_bw_gradients.append(angle)
            lg.param_norms.append(param_norms.numpy())
            lg.train_loss.append(train_loss.numpy())

            temp_train_loss, train_accuracy = model.evaluate(dataset_train, verbose=False)
            lg.train_loss.append(temp_train_loss)
            lg.train_accuracy.append(train_accuracy)

            test_loss, test_accuracy = model.evaluate(dataset_test, verbose=False)
            lg.test_loss.append(test_loss)
            lg.test_accuracy.append(test_accuracy)
            if cfg.SHOW_PROGRESS:
                print('Test Loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy))

            # calculate rejection i.e. (remove the component of regularization gradient along training loss gradient)
            # this is refered to as predictor direction in the paper

            # if typical entries in the training loss gradient (RMS values) are greater than a certain near-zero
            # threshold, we will likely not run into numerical instability issues
            if tf.sqrt(tf.reduce_mean(tf.square(grad_loss))).numpy() >= (
            1e-6):  # numerical stability issues with vector projection calculations (denominator)
                rejection_pred = get_rejection(grad_param_norms, grad_loss)
            else:
                rejection_pred = grad_param_norms

            # normalizing seems to be giving good results
            rejection_pred_normalized = rejection_pred / np.linalg.norm(rejection_pred)

            # now extract from the big rejection vector pieces that correspond to different weight vectors/matrices
            list_grads = recover_individual_grads(rejection_pred_normalized, model)

            if last == 'predict':
                # the last step was a predictor-type, we calcualte the angle between the last predicted direction and the current one
                temp_angle = get_angle(rejection_pred_normalized, rejection_pred_prev, unitvecs=True) * 180 / np.pi
                buffer_direction_change_pred.append(temp_angle.numpy())
                buffer_direction_change_pred = buffer_direction_change_pred[
                                               -cfg.NUM_TRACKED_DIRECTION_CHANGES:]  # fixed-size buffer?

            elif last == 'correct':
                buffer_direction_change_pred = []
                lr_pred_curr = lr_pred_min  # re-initialize the predictor learning rate with its min

            if len(buffer_direction_change_pred) >= cfg.NUM_TRACKED_DIRECTION_CHANGES:
                if cfg.SHOW_PROGRESS:
                    print('here', np.mean(buffer_direction_change_pred), np.mean(np.abs(buffer_direction_change_pred)))
                avg_change_direction = np.mean(np.abs(buffer_direction_change_pred))  # note the abs!

                if last == 'predict':

                    if avg_change_direction >= cfg.THRESHOLD_DEGS_DIRECTION_CHANGES:
                        string_direction_change = 'Angle b/w consecutive predicted directions >= {} degrees'
                        lr_pred_curr = max(round(lr_pred_curr * 0.1, 10), lr_pred_min)

                    else:
                        string_direction_change = 'Angle b/w consecutive predicted directions < {} degrees'
                        lr_pred_curr = min(round(lr_pred_curr * 1.1, 10), lr_pred_max)

                    if cfg.SHOW_PROGRESS:
                        print(string_direction_change.format(cfg.THRESHOLD_DEGS_DIRECTION_CHANGES))

            if cfg.SHOW_PROGRESS:
                string_new_lr = "New lr(pred): {}, New lr(corr): {}"
                #print(string_new_lr.format(lr_pred_curr, lr_corr_curr))

            lg.lr_pred.append(lr_pred_curr)
            lg.index_epoch_pred.append(epoch)

            # finally, update weights
            model.optimizer.lr.assign(lr_pred_curr)
            model.optimizer.apply_gradients(zip(list_grads, model.trainable_variables))

            # update step-type flag
            last = 'predict'
            rejection_pred_prev = rejection_pred_normalized

            # Saving

            if  (epoch_pred == 1) or (epoch_pred % 50 == 0):
                all_weights = np.hstack([var.numpy().ravel() for var in model.trainable_variables])
                lg.weights_phase2.append(all_weights)


        else:
            if cfg.SHOW_PROGRESS:
                print('$$$ CORRECTION $$$')
            # need to constrain gradients w.r.t loss_change to a direction perpendicular to it!

            arrays_grad_loss_change = [tape.gradient(loss_change, p) for p in model.trainable_variables]
            grad_loss_change = tf.concat([tf.reshape(tv, [-1]) for tv in arrays_grad_loss_change], axis=0)

            try:
                rejection_corr = get_rejection(grad_loss_change, rejection_pred_prev, unit_b=True)
            except:
                rejection_corr = grad_loss_change

            rejection_corr_normalized = rejection_corr / tf.linalg.norm(rejection_corr)

            # recover and apply gradients/slices for individual weights
            list_grads = recover_individual_grads(rejection_corr_normalized, model)

            if last == 'correct':
                temp_angle = get_angle(rejection_corr_normalized, rejection_corr_prev, unitvecs=True) * 180 / np.pi
                buffer_direction_change_corr.append(temp_angle.numpy())
                buffer_direction_change_corr = buffer_direction_change_corr[
                                               -cfg.NUM_TRACKED_DIRECTION_CHANGES:]  # fixed-size buffer?

            elif last == 'predict':
                buffer_direction_change_corr = []
                lr_corr_curr = lr_corr_min

                if len(buffer_direction_change_corr) >= cfg.NUM_TRACKED_DIRECTION_CHANGES:
                    avg_change_direction = np.mean(np.abs(buffer_direction_change_corr))  # note the abs!

                    if last == 'correct':
                        if avg_change_direction >= cfg.THRESHOLD_DEGS_DIRECTION_CHANGES:

                            string_direction_change = 'Angle b/w consecutive corrector directions >= {} degrees'
                            lr_corr_curr = max(round(lr_corr_curr * 0.1, 10), lr_corr_min)

                        else:

                            string_direction_change = 'Angle b/w consecutive corrector directions < {} degrees'
                            lr_corr_curr = min(round(lr_corr_curr * 1.1, 10), lr_corr_max)

                        if cfg.SHOW_PROGRESS:
                            print(string_direction_change.format(cfg.THRESHOLD_DEGS_DIRECTION_CHANGES))

                if cfg.SHOW_PROGRESS:
                    string_new_lr = "New lr(pred): {}, New lr(corr): {}"
                    print(string_new_lr.format(lr_pred_curr, lr_corr_curr))

            lg.lr_corr.append(lr_corr_curr)
            lg.index_epoch_corr.append(epoch)

            model.optimizer.lr.assign(lr_corr_curr)
            model.optimizer.apply_gradients(zip(list_grads, model.trainable_variables))

            # update step-type flag
            last = 'correct'
            rejection_corr_prev = rejection_corr_normalized

        del tape, list_grads

        loop_condition = ((not lg.angle_bw_gradients) or (lg.angle_bw_gradients[-1] < 180.0)) and (
                    epoch_pred < cfg.NUM_EPOCHS_PHASE2)

        if epoch % 1000 == 0:
            #clear_output(wait=True)
            print('EPOCH #: {}\n\n'.format(epoch))

    print('Phase 2 is done!')

    names_logs_to_save = [att for att in dir(lg) if '__' not in att]
    logs_to_save = [getattr(lg, temp) for temp in names_logs_to_save]
    dict_logs_to_save = dict(zip(names_logs_to_save, logs_to_save))
    temp_path = os.path.join(cfg.NAME_EXP, 'logs_{}_exp_{}').format(cfg.NAME_EXP, exp_idx)
    np.savez(temp_path, **dict_logs_to_save)

