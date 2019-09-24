# -*- coding: utf-8 -*-
"""Helper functions."""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os
import pickle

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


class CrossValidator:

    def __init__(self,
                 task,
                 data_mode,
                 results_dir,
                 model_name,
                 epochs,
                 train_generator,
                 test_generator,
                 t,
                 k,
                 channel_drop=False,
                 np_random_state=71):
        assert task in ('rnr', 'hmdd'), "task must be one of {'rnr', 'hmdd'}"
        assert data_mode in ('cross_subject', 'balanced'), "data_mode must be one of {'cross_subject', 'balanced'}"

        self.task = task
        self.data_mode = data_mode
        self.results_dir = results_dir
        self.model_name = model_name
        self.epochs = epochs
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.t = t
        self.k = k
        self.channel_drop = channel_drop
        self.np_random_state = np_random_state

        if hasattr(train_generator, 'min_duration'):
            train_gen_type = 'var'
            tr_pr_1 = train_generator.min_duration
            tr_pr_2 = train_generator.max_duration
        else:
            train_gen_type = 'fixed'
            tr_pr_1 = train_generator.duration
            tr_pr_2 = train_generator.overlap
        if hasattr(test_generator, 'min_duration'):
            test_gen_type = 'var'
            te_pr_1 = test_generator.min_duration
            te_pr_2 = test_generator.max_duration
        else:
            test_gen_type = 'fixed'
            te_pr_1 = test_generator.duration
            te_pr_2 = test_generator.overlap

        train_data_prefix = train_gen_type + str(tr_pr_1) + str(tr_pr_2)
        test_data_prefix = test_gen_type + str(te_pr_1) + str(te_pr_2)

        self.cv_dir = os.path.join(results_dir, '{}_{}'.format(data_mode, task))
        if not os.path.exists(self.cv_dir):
            os.mkdir(self.cv_dir)

        unique_identifier = '{}time-{}fold-{}epochs-tr_{}-te_{}'.format(t,
                                                                        k,
                                                                        epochs,
                                                                        train_data_prefix,
                                                                        test_data_prefix)
        indices_filename = 'train_test_indices-{}.pkl'.format(unique_identifier)
        self.indices_path = os.path.join(self.cv_dir, indices_filename)

        scores_filename = '{}-{}.npy'.format(model_name, unique_identifier)
        self.scores_path = os.path.join(results_dir, scores_filename)
        self.rounds_file_names = ['{}-time{}-fold{}-{}epochs-tr_{}-te_{}.npy'.format(model_name,
                                                                                     i + 1,
                                                                                     j + 1,
                                                                                     epochs,
                                                                                     train_data_prefix,
                                                                                     test_data_prefix) for i in range(t) for j in range(k)]
        self.rounds_file_paths = [os.path.join(results_dir, file_name) for file_name in self.rounds_file_names]

    def do_cv(self,
              model_obj,
              data,
              labels):
        if os.path.exists(self.scores_path):
            print('Final scores already exists.')
            final_scores = np.load(self.scores_path)
            return final_scores

        train_indices, test_indices = self._get_train_test_indices(data, labels)
        dir_file_names = os.listdir(self.cv_dir)
        for i in range(self.t):
            print('time {}/{}:'.format(i + 1, self.t))
            for j in range(self.k):
                print(' step {}/{} ...'.format(j + 1, self.k))
                ind = int(i * self.t + j)
                file_name = self.rounds_file_names[ind]
                file_path = self.rounds_file_paths[ind]
                if file_name not in dir_file_names:
                    train_ind = train_indices[i][j]
                    test_ind = test_indices[i][j]
                    scores = self._do_train_eval(model_obj,
                                                 data,
                                                 labels,
                                                 train_ind,
                                                 test_ind)
                    np.save(file_path, scores)
        final_scores = self._generate_final_scores()
        return final_scores

    def _get_train_test_indices(self, data, labels):
        if os.path.exists(self.indices_path):
            with open(self.indices_path, 'rb') as pkl:
                indices = pickle.load(pkl)
            train_indices = indices[0]
            test_indices = indices[1]
            print('Train-test indices already exists.')
        else:
            train_indices = list()
            test_indices = list()
            for i in range(self.t):
                train_indices.append(list())
                test_indices.append(list())
                folds = StratifiedKFold(n_splits=self.k,
                                        shuffle=True,
                                        random_state=self.np_random_state)
                for train_ind, test_ind in folds.split(data, labels):
                    train_indices[-1].append(train_ind)
                    test_indices[-1].append(test_ind)
            with open(self.indices_path, 'wb') as pkl:
                pickle.dump([train_indices, test_indices], pkl)
            print('Train-test indices generated.')
        return train_indices, test_indices

    def _do_train_eval(self,
                       model_obj,
                       data,
                       labels,
                       train_ind,
                       test_ind):
        loss = model_obj.loss
        optimizer = model_obj.optimizer
        metrics = model_obj.metrics

        if self.data_mode == 'cross_subject':
            train_data = [data[j] for j in train_ind]
            train_labels = [labels[j] for j in train_ind]
            test_data = [data[j] for j in test_ind]
            test_labels = [labels[j] for j in test_ind]
            train_gen, n_iter_train = self.train_generator.get_generator(train_data, train_labels)
            test_gen, n_iter_test = self.train_generator.get_generator(test_data, test_labels)
        else:
            train_gen, n_iter_train = self.test_generator.get_generator(data, labels, train_ind)
            test_gen, n_iter_test = self.test_generator.get_generator(data, labels, test_ind)

        model = model_obj.create_model()
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=n_iter_train,
                            epochs=self.epochs,
                            verbose=False)
        scores = model.evaluate_generator(test_gen,
                                          steps=n_iter_test,
                                          verbose=False)
        if self.channel_drop:
            scores = [scores]
            scores.extend(self._get_channel_drop_scores(test_gen,
                                                        n_iter_test,
                                                        model))

        return scores

    def _generate_final_scores(self):
        final_scores = list()
        for file_path in self.rounds_file_paths:
            final_scores.append(np.load(file_path))
        final_scores = np.array(final_scores).reshape((self.t, self.k, final_scores[0].shape[0]))
        np.save(self.scores_path, final_scores)
        for file_path in self.rounds_file_paths:
            os.remove(file_path)
        return final_scores

    def _get_channel_drop_scores(self,
                                 test_gen,
                                 n_iter_test,
                                 model):
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
                x_dropped = self.drop_channels(x_test, drop ** 2)
            y_prob = model.predict(x_dropped)[:, 0]
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            fpr.append(false_positive_rate)
            tpr.append(true_positive_rate)
            th.append(thresholds)
            rocauc.append(roc_auc)
        return np.array([fpr, tpr, th, rocauc])

    @staticmethod
    def drop_channels(arr, drop=2):
        n_samples, n_times, n_channels = arr.shape
        to_drop = np.random.randint(low=0, high=n_channels, size=(n_samples, drop))
        dropped_x = arr.copy()
        for i, channels in enumerate(to_drop):
            dropped_x[i, :, channels] = 0
        return dropped_x


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
