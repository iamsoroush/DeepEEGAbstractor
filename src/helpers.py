# -*- coding: utf-8 -*-
"""Helper functions."""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os
from .dataset import data_generator

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

if tf.__version__.startswith('2'):
    from tensorflow import keras
else:
    import keras


def kfold_cv(model_obj,
             data,
             labels,
             k=4,
             batch_size=64,
             epochs=20):
    scores = list()
    folds = StratifiedKFold(n_splits=k, shuffle=True)
    loss = model_obj.loss
    optimizer = model_obj.optimizer
    metrics = model_obj.metrics
    for i, (train_index, test_index) in enumerate(folds.split(np.zeros(len(data)), labels)):
        model = model_obj.create_model()
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        train_gen = data_generator(data, labels, train_index.copy(), batch_size)
        test_gen = data_generator(data, labels, test_index.copy(), batch_size)
        n_iter_train = len(train_index) // batch_size
        n_iter_test = len(test_index) // batch_size
        print(' step {}/{} ...'.format(i + 1, k))
        model.fit_generator(generator=train_gen, steps_per_epoch=n_iter_train,
                            epochs=epochs, verbose=False)
        score = model.evaluate_generator(test_gen, steps=n_iter_test, verbose=False)
        scores.append(score)
    scores = np.array(scores)
    return scores


def ttime_kfold_cv(model_obj,
                   data,
                   labels,
                   t=5,
                   k=2,
                   batch_size=64,
                   epochs=20):
    scores = list()
    for i in range(t):
        print('time {}:'.format(i + 1))
        score = kfold_cv(model_obj, data, labels, k, batch_size, epochs)
        scores.append(score)
    scores = np.array(scores)
    return scores


def ttime_kfold_cross_validation(model_obj,
                                 data,
                                 labels,
                                 train_indices,
                                 test_indices,
                                 generator,
                                 t,
                                 k,
                                 epochs,
                                 results_dir,
                                 task,
                                 instance_duration,
                                 data_mode):
    # train_indices: [[t1_k1_ind, t1_k2_ind, ...], [t2_k1_ind, ...], ...]
    model_name = model_obj.model_name_
    scores_filename = '{}-{}-{}t-{}k-{}-duration{}.npy'.format(model_name,
                                                               task,
                                                               t,
                                                               k,
                                                               'cross_subject',
                                                               instance_duration)
    scores_path = os.path.join(results_dir, scores_filename)
    if os.path.exists(scores_path):
        print('Final scores already exists.')
        final_scores = np.load(scores_path)
        return final_scores

    file_names = ['{}-{}-time{}-fold{}-cv.npy'.format(model_name,
                                                      task,
                                                      i + 1,
                                                      j + 1) for i in range(t) for j in range(k)]
    file_paths = [os.path.join(results_dir, file_name) for file_name in file_names]
    dir_file_names = os.listdir(results_dir)
    for i in range(t):
        print('time {}:'.format(i + 1))
        for j in range(k):
            print(' step {}/{} ...'.format(j + 1, k))
            ind = int(i * t + j)
            file_name = file_names[ind]
            file_path = file_paths[ind]
            if file_name not in dir_file_names:
                train_ind = train_indices[i][j]
                test_ind = test_indices[i][j]
                if data_mode == 'cross_subject':
                    scores = _do_train_eval(model_obj,
                                            data,
                                            labels,
                                            train_ind,
                                            test_ind,
                                            generator,
                                            epochs)
                else:
                    scores = _do_train_eval_balanced(model_obj,
                                                     data,
                                                     labels,
                                                     train_ind,
                                                     test_ind,
                                                     generator,
                                                     epochs)
                np.save(file_path, scores)
    final_scores = list()
    for file_path in file_paths:
        final_scores.append(np.load(file_path))
    final_scores = np.array(final_scores).reshape((t, k, final_scores[0].shape[0]))
    np.save(scores_path, final_scores)
    for file_path in file_paths:
        os.remove(file_path)
    return final_scores


def _do_train_eval(model_obj,
                   data,
                   labels,
                   train_ind,
                   test_ind,
                   generator,
                   epochs):
    loss = model_obj.loss
    optimizer = model_obj.optimizer
    metrics = model_obj.metrics

    train_data = [data[j] for j in train_ind]
    train_labels = [labels[j] for j in train_ind]
    test_data = [data[j] for j in test_ind]
    test_labels = [labels[j] for j in test_ind]
    train_gen, n_iter_train = generator.get_generator(train_data, train_labels)
    test_gen, n_iter_test = generator.get_generator(test_data, test_labels)

    model = model_obj.create_model()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=n_iter_train,
                        epochs=epochs,
                        verbose=False)
    scores = model.evaluate_generator(test_gen, steps=n_iter_test, verbose=False)
    return scores


def _do_train_eval_balanced(model_obj,
                            data,
                            labels,
                            train_ind,
                            test_ind,
                            generator,
                            epochs):
    loss = model_obj.loss
    optimizer = model_obj.optimizer
    metrics = model_obj.metrics

    train_gen, n_iter_train = generator.get_generator(data, labels, train_ind)
    test_gen, n_iter_test = generator.get_generator(data, labels, test_ind)

    model = model_obj.create_model()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=n_iter_train,
                        epochs=epochs,
                        verbose=False)
    scores = model.evaluate_generator(test_gen, steps=n_iter_test, verbose=False)
    return scores


def ttime_kfold_cross_validation_v2(model_obj,
                                    data,
                                    labels,
                                    train_indices,
                                    test_indices,
                                    generator,
                                    t,
                                    k,
                                    epochs,
                                    results_dir,
                                    task,
                                    instance_duration):
    # train_indices: [[t1_k1_ind, t1_k2_ind, ...], [t2_k1_ind, ...], ...]
    model_name = model_obj.model_name_
    scores_filename = '{}-{}-{}t-{}k-{}-duration{}.npy'.format(model_name,
                                                               task,
                                                               t,
                                                               k,
                                                               'cross_subject',
                                                               instance_duration)
    scores_path = os.path.join(results_dir, scores_filename)
    if os.path.exists(scores_path):
        print('Final scores already exists.')
        final_scores = np.load(scores_path, allow_pickle=True)
        return final_scores

    file_names = ['{}-{}-time{}-fold{}-cv.npy'.format(model_name,
                                                      task,
                                                      i + 1,
                                                      j + 1) for i in range(t) for j in range(k)]
    file_paths = [os.path.join(results_dir, file_name) for file_name in file_names]
    dir_file_names = os.listdir(results_dir)
    for i in range(t):
        print('time {}:'.format(i + 1))
        for j in range(k):
            print(' step {}/{} ...'.format(j + 1, k))
            ind = int(i * t + j)
            file_name = file_names[ind]
            file_path = file_paths[ind]
            if file_name not in dir_file_names:
                train_ind = train_indices[i][j]
                test_ind = test_indices[i][j]
                scores = _do_train_eval_balanced_v2(model_obj,
                                                    data,
                                                    labels,
                                                    train_ind,
                                                    test_ind,
                                                    generator,
                                                    epochs)
                np.save(file_path, scores)

    final_scores = list()
    for file_path in file_paths:
        final_scores.append(np.load(file_path, allow_pickle=True))
    final_scores = np.array(final_scores).reshape((t, k, final_scores[0].shape[0]))
    np.save(scores_path, final_scores)
    for file_path in file_paths:
        os.remove(file_path)
    return final_scores


def _do_train_eval_balanced_v2(model_obj,
                               data,
                               labels,
                               train_ind,
                               test_ind,
                               generator,
                               epochs):
    loss = model_obj.loss
    optimizer = model_obj.optimizer
    metrics = model_obj.metrics

    train_gen, n_iter_train = generator.get_generator(data, labels, train_ind)
    test_gen, n_iter_test = generator.get_generator(data, labels, test_ind)

    model = model_obj.create_model()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=n_iter_train,
                        epochs=epochs,
                        verbose=False)
    scores = model.evaluate_generator(test_gen, steps=n_iter_test, verbose=False)
    x_test, y_test = list(), list()
    for i in range(n_iter_test):
        x_batch, y_batch = next(test_gen)
        x_test.extend(x_batch)
        y_test.extend(y_batch)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    fpr = list()
    tpr = list()
    th = list()
    rocauc = list()
    for drop in range(4):
        if drop == 0:
            x_dropped = x_test
        else:
            x_dropped = drop_channels(x_test, drop ** 2)
        y_prob = model.predict(x_dropped)[:, 0]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        fpr.append(false_positive_rate)
        tpr.append(true_positive_rate)
        th.append(thresholds)
        rocauc.append(roc_auc)

    return np.array([scores, fpr, tpr, th, rocauc])


def ttime_kfold_cross_validation_v3(model_obj,
                                    data,
                                    labels,
                                    train_indices,
                                    test_indices,
                                    train_generator,
                                    test_generator,
                                    t,
                                    k,
                                    epochs,
                                    results_dir,
                                    task,
                                    instance_duration):
    # train_indices: [[t1_k1_ind, t1_k2_ind, ...], [t2_k1_ind, ...], ...]
    model_name = model_obj.model_name_
    scores_filename = '{}-{}-{}t-{}k-{}-duration{}.npy'.format(model_name,
                                                               task,
                                                               t,
                                                               k,
                                                               'cross_subject',
                                                               instance_duration)
    scores_path = os.path.join(results_dir, scores_filename)
    if os.path.exists(scores_path):
        print('Final scores already exists.')
        final_scores = np.load(scores_path, allow_pickle=True)
        return final_scores

    file_names = ['{}-{}-time{}-fold{}-cv.npy'.format(model_name,
                                                      task,
                                                      i + 1,
                                                      j + 1) for i in range(t) for j in range(k)]
    file_paths = [os.path.join(results_dir, file_name) for file_name in file_names]
    dir_file_names = os.listdir(results_dir)
    for i in range(t):
        print('time {}:'.format(i + 1))
        for j in range(k):
            print(' step {}/{} ...'.format(j + 1, k))
            ind = int(i * t + j)
            file_name = file_names[ind]
            file_path = file_paths[ind]
            if file_name not in dir_file_names:
                train_ind = train_indices[i][j]
                test_ind = test_indices[i][j]
                scores = _do_train_eval_v3(model_obj,
                                           data,
                                           labels,
                                           train_ind,
                                           test_ind,
                                           train_generator,
                                           test_generator,
                                           epochs)
                np.save(file_path, scores)

    final_scores = list()
    for file_path in file_paths:
        final_scores.append(np.load(file_path, allow_pickle=True))
    final_scores = np.array(final_scores).reshape((t, k, final_scores[0].shape[0]))
    np.save(scores_path, final_scores)
    for file_path in file_paths:
        os.remove(file_path)
    return final_scores


def _do_train_eval_v3(model_obj,
                      data,
                      labels,
                      train_ind,
                      test_ind,
                      train_generator,
                      test_generator,
                      epochs):
    loss = model_obj.loss
    optimizer = model_obj.optimizer
    metrics = model_obj.metrics

    train_data = [data[j] for j in train_ind]
    train_labels = [labels[j] for j in train_ind]
    test_data = [data[j] for j in test_ind]
    test_labels = [labels[j] for j in test_ind]
    train_gen, n_iter_train = train_generator.get_generator(train_data, train_labels)
    test_gen, n_iter_test = test_generator.get_generator(test_data, test_labels)

    model = model_obj.create_model()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=n_iter_train,
                        epochs=epochs,
                        verbose=False)
    scores = model.evaluate_generator(test_gen, steps=n_iter_test, verbose=False)
    return scores


def drop_channels(arr, drop=2):
    n_samples, n_times, n_channels = arr.shape
    to_drop = np.random.randint(low=0, high=n_channels, size=(n_samples, drop))
    dropped_x = arr.copy()
    for i, channels in enumerate(to_drop):
        dropped_x[i, :, channels] = 0
    return dropped_x


def plot_scores(scores,
                selected_model,
                t,
                k,
                task,
                instance_duration,
                data_mode,
                cv_results_dir,
                dpi=96):
    keys = list()
    fig, ax = plt.subplots(figsize=(20, 8), dpi=dpi)

    x_coord = 0.8
    y_coord = 0.02
    for key, value in scores.items():
        linewidth = 1
        alpha = 0.6
        if key == 'binary_accuracy':
            linewidth = 2
            alpha = 0.8
            ax.plot(value, linewidth=linewidth, marker='o', alpha=alpha)
        else:
            ax.plot(value, linewidth=linewidth, alpha=alpha)
        mean = value.mean()
        std = value.std(ddof=1)
        ax.text(x_coord, y_coord, '{}: {:2.2f} +- {:2.2f}'.format(key,
                                                                  mean * 100,
                                                                  std * 100),
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes)
        y_coord += 0.03
        keys.append(key)

    ax.legend(keys, loc='lower left')
    ax.set_title(selected_model)
    ax.set_xlabel('# Round')
    # ax.set_xticks(range(1, t * k + 1),  direction='vertical')
    ax.set_ylabel('Score')
    # ax.set_ylim(max(0, min_score - 0.2), 1)
    plot_name = '{}-{}-{}t-{}k-{}-duration{}.jpg'.format(selected_model,
                                                         task,
                                                         t,
                                                         k,
                                                         data_mode,
                                                         instance_duration)
    path_to_save = os.path.join(cv_results_dir, plot_name)
    fig.savefig(path_to_save)


def t_paired_test():
    # TODO: define this function
    pass


def f1_score(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    recall = true_positives / (possible_positives + keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())
    return f1_val


def sensitivity(y_true, y_pred):
    # recall: true_p / possible_p
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + keras.backend.epsilon())


def specificity(y_true, y_pred):
    # true_n / possible_n
    true_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + keras.backend.epsilon())
