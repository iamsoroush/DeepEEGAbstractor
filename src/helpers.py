# -*- coding: utf-8 -*-
"""Helper functions."""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os
import pickle
from itertools import combinations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, mean_squared_error
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

from . import keras

plt.style.use('seaborn-darkgrid')


class CrossValidator:

    def __init__(self,
                 task,
                 data_mode,
                 main_res_dir,
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
        self.main_res_dir = main_res_dir
        self.model_name = model_name
        self.epochs = epochs
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.t = t
        self.k = k
        self.channel_drop = channel_drop
        self.np_random_state = np_random_state
        self.cv_dir = None
        self.indices_path = None
        self.scores_path = None
        self.rounds_file_names = None
        self.rounds_file_paths = None
        self._initialize()

    def _initialize(self):
        if not self.train_generator.is_fixed:
            train_gen_type = 'var'
            tr_pr_1 = self.train_generator.min_duration
            tr_pr_2 = self.train_generator.max_duration
        else:
            train_gen_type = 'fixed'
            tr_pr_1 = self.train_generator.duration
            tr_pr_2 = self.train_generator.overlap
        if not self.test_generator.is_fixed:
            test_gen_type = 'var'
            te_pr_1 = self.test_generator.min_duration
            te_pr_2 = self.test_generator.max_duration
        else:
            test_gen_type = 'fixed'
            te_pr_1 = self.test_generator.duration
            te_pr_2 = self.test_generator.overlap

        train_data_prefix = train_gen_type + str(tr_pr_1) + str(tr_pr_2)
        test_data_prefix = test_gen_type + str(te_pr_1) + str(te_pr_2)

        self.cv_dir = os.path.join(self.main_res_dir, '{}_{}'.format(self.data_mode, self.task))
        if not os.path.exists(self.cv_dir):
            os.mkdir(self.cv_dir)

        unique_identifier = '{}time-{}fold-{}epochs-tr_{}-te_{}'.format(self.t,
                                                                        self.k,
                                                                        self.epochs,
                                                                        train_data_prefix,
                                                                        test_data_prefix)
        indices_filename = 'train_test_indices-{}.pkl'.format(unique_identifier)
        self.indices_path = os.path.join(self.cv_dir, indices_filename)

        scores_filename = '{}-{}.npy'.format(self.model_name, unique_identifier)
        self.scores_path = os.path.join(self.cv_dir, scores_filename)
        template = '{}-time{}-fold{}-{}epochs-tr_{}-te_{}.npy'
        self.rounds_file_names = [template.format(self.model_name,
                                                  i + 1,
                                                  j + 1,
                                                  self.epochs,
                                                  train_data_prefix,
                                                  test_data_prefix) for i in range(self.t) for j in range(self.k)]
        self.rounds_file_paths = [os.path.join(self.cv_dir, file_name) for file_name in self.rounds_file_names]
        return

    def do_cv(self,
              model_obj,
              data,
              labels):
        if os.path.exists(self.scores_path):
            print('Final scores already exists.')
            final_scores = np.load(self.scores_path, allow_pickle=True)
            return final_scores

        train_indices, test_indices = self._get_train_test_indices(data, labels)
        dir_file_names = os.listdir(self.cv_dir)
        for i in range(self.t):
            print('time {}/{}:'.format(i + 1, self.t))
            for j in range(self.k):
                print(' step {}/{} ...'.format(j + 1, self.k))
                ind = int(i * self.k + j)
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
        self.plot_channel_drop_roc()
        self.plot_scores()
        self.plot_subject_wise_scores()
        return final_scores

    def plot_scores(self, dpi=80):
        if not os.path.exists(self.scores_path):
            print('Final scores does not exist.')
            return

        scores = np.array(list(np.load(self.scores_path, allow_pickle=True)[:, 0]))
        keys = ['MSE Loss', 'Accuracy', 'F1-Score', 'Sensitivity', 'Specificity']
        fig, ax = plt.subplots(figsize=(20, 8), dpi=dpi)

        x_coord = 0.8
        y_coord = 0.02
        for key, values in zip(keys, scores.T):
            linewidth = 1
            alpha = 0.6
            if key == 'Accuracy':
                linewidth = 2
                alpha = 0.8
                ax.plot(values, linewidth=linewidth, marker='o', alpha=alpha)
            elif key == 'MSE Loss':
                pass
            else:
                ax.plot(values, linewidth=linewidth, alpha=alpha)
            mean = values.mean()
            std = values.std(ddof=1)
            ax.text(x_coord, y_coord, '{}: {:2.3f} +- {:2.3f}'.format(key,
                                                                      mean,
                                                                      std),
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes)
            y_coord += 0.03
            keys.append(key)

        ax.legend(keys[1:], loc='lower left')
        ax.set_title(self.model_name)
        ax.set_xlabel('# Round')
        # ax.set_xticks(range(1, t * k + 1),  direction='vertical')
        ax.set_ylabel('Score')
        # ax.set_ylim(max(0, min_score - 0.2), 1)
        plot_name = '{}.jpg'.format(os.path.basename(self.scores_path).split('.')[0])
        path_to_save = os.path.join(os.path.dirname(self.scores_path), plot_name)
        fig.savefig(path_to_save)

    def plot_subject_wise_scores(self, dpi=80):
        if not os.path.exists(self.scores_path):
            print('Final scores does not exist.')
            return
        scores = np.array(list(np.load(self.scores_path, allow_pickle=True)[:, 1]))
        tns = scores[:, 0]
        fps = scores[:, 1]
        fns = scores[:, 2]
        tps = scores[:, 3]
        acc_vector = (tps + tns) / (tps + fns + fps + tns)
        prec_vector = tps / (tps + fps + 0.001)
        rec_vector = tps / (tps + fns + 0.001)
        spec_vector = tns / (tns + fps + 0.001)
        fscore_vector = 2 * (prec_vector * rec_vector) / (prec_vector + rec_vector + 0.0001)

        keys = ['Accuracy', 'F1-Score', 'Sensitivity', 'Specificity']
        fig, ax = plt.subplots(figsize=(20, 8), dpi=dpi)

        x_coord = 0.8
        y_coord = 0.02
        for key, values in zip(keys, [acc_vector, fscore_vector, rec_vector, spec_vector]):
            linewidth = 1
            alpha = 0.6
            if key == 'Accuracy':
                linewidth = 2
                alpha = 0.8
                ax.plot(values, linewidth=linewidth, marker='o', alpha=alpha)
            else:
                ax.plot(values, linewidth=linewidth, alpha=alpha)
            mean = values.mean()
            std = values.std(ddof=1)
            ax.text(x_coord, y_coord, '{}: {:2.3f} +- {:2.3f}'.format(key,
                                                                      mean,
                                                                      std),
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes)
            y_coord += 0.03
            keys.append(key)

        ax.legend(keys, loc='lower left')
        ax.set_title(self.model_name)
        ax.set_xlabel('# Round')
        # ax.set_xticks(range(1, t * k + 1),  direction='vertical')
        ax.set_ylabel('Score')
        # ax.set_ylim(max(0, min_score - 0.2), 1)
        plot_name = '{}_subject-wise-scores.jpg'.format(os.path.basename(self.scores_path).split('.')[0])
        path_to_save = os.path.join(os.path.dirname(self.scores_path), plot_name)
        fig.savefig(path_to_save)

    def plot_channel_drop_roc(self):
        if not os.path.exists(self.scores_path):
            print('Final scores does not exist.')
            return
        scores = np.array(list(np.load(self.scores_path, allow_pickle=True)[:, 2]))
        fprs = np.array(scores[:, 0])
        tprs = np.array(scores[:, 1])

        self._roc_vs_channel_drop(fprs,
                                  tprs)

    def _roc_vs_channel_drop(self,
                             fprs,
                             tprs):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
        for i, ax in enumerate(axes.flatten()):
            fp = fprs[:, i]
            tp = tprs[:, i]
            self._draw_roc_curve(fp, tp, ax)
            ax.set_title('# Channels Dropped: {}'.format(i ** 2))

        fig.suptitle('Model: {}, Task: "{}"'.format(self.model_name, self.task))

        plot_name = '{}_channel-drop.jpg'.format(os.path.basename(self.scores_path).split('.')[0])
        path_to_save = os.path.join(os.path.dirname(self.scores_path), plot_name)
        fig.savefig(path_to_save)

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
                                        random_state=(i + 1) * self.np_random_state)
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
        """Doing one training-validation step in kfold cross validation.

        At the end, saves a numpy array:
            [[loss, binary_accuracy, f1_score, sensitivity, specificity],
             [subject-wise_TN, subject-wise_FP, subject-wise_FN, subject-wise_TP],
             [ch-drop-fpr, ch-drop-tpr, ch-drop-th, ch-drop-roc-auc]]
        """
        loss = model_obj.loss
        optimizer = model_obj.optimizer
        # metrics = model_obj.metrics

        if self.data_mode == 'cross_subject':
            train_data = [data[j] for j in train_ind]
            train_labels = [labels[j] for j in train_ind]
            test_data = [data[j] for j in test_ind]
            test_labels = [labels[j] for j in test_ind]
            train_gen, n_iter_train = self.train_generator.get_generator(data=train_data,
                                                                         labels=train_labels)
            test_gen, n_iter_test = self.test_generator.get_generator(data=test_data,
                                                                      labels=test_labels)
        else:
            train_gen, n_iter_train = self.train_generator.get_generator(data=data,
                                                                         labels=labels,
                                                                         indxs=train_ind)
            test_gen, n_iter_test = self.test_generator.get_generator(data=data,
                                                                      labels=labels,
                                                                      indxs=test_ind)

        es_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=5)
        model = model_obj.create_model()
        model.compile(loss=loss, optimizer=optimizer)

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=n_iter_train,
                            epochs=self.epochs,
                            verbose=False,
                            callbacks=[es_callback])

        scores = [list() for _ in range(3)]
        x_test = list()
        y_test = list()
        for i in range(n_iter_test):
            x_batch, y_batch = next(test_gen)
            x_test.extend(x_batch)
            y_test.extend(y_batch)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        y_pred = model.predict(x_test)
        scores[0].append(mean_squared_error(y_test, y_pred))
        y_pred = np.where(y_pred > 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        scores[0].append((tp + tn) / (tp + tn + fp + fn))
        scores[0].append(f1_score(y_test, y_pred))
        scores[0].append(tp / (tp + fn))
        scores[0].append(tn / (tn + fp))

        if self.data_mode == 'cross_subject':
            scores[1].extend(self._calc_subject_wise_scores(model,
                                                            x_test,
                                                            y_test))

        if self.channel_drop:
            scores[2].extend(self._get_channel_drop_scores(x_test,
                                                           y_test,
                                                           model))
        return np.array(scores)

    def _calc_subject_wise_scores(self,
                                  model,
                                  x_test,
                                  y_test):
        subject_ids = np.array(self.test_generator.belonged_to_subject[: len(y_test)])
        y_subjects = list()
        y_preds = list()
        for s_id in np.unique(subject_ids):
            indx = np.where(subject_ids == s_id)[0]
            x_subject = x_test[indx]
            y_subjects.append(int(y_test[indx][0]))
            y_pred_proba = model.predict(x_subject).mean()
            y_preds.append(int(np.where(y_pred_proba > 0.5, 1, 0)))
            # y_pred_proba = model.predict(x_subject)
            # y_pred = np.where(y_pred_proba > 0.5, 1, 0).mean()
            # if y_pred >= 0.5:
            #     y_preds.append(1)
            # else:
            #     y_preds.append(0)
        tn, fp, fn, tp = confusion_matrix(y_subjects, y_preds).ravel()
        return np.array([tn, fp, fn, tp])

    def _generate_final_scores(self):
        final_scores = list()
        for file_path in self.rounds_file_paths:
            final_scores.append(np.load(file_path, allow_pickle=True))
        final_scores = np.array(final_scores)
        np.save(self.scores_path, final_scores)
        for file_path in self.rounds_file_paths:
            os.remove(file_path)
        return final_scores

    def _get_channel_drop_scores(self,
                                 x_test,
                                 y_test,
                                 model):
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
        return [fpr, tpr, th, rocauc]

    @staticmethod
    def _draw_roc_curve(fps, tps, ax):
        roc_auc = list()
        for i, j in zip(fps, tps):
            roc_auc.append(auc(i, j))
            linewidth = 1
            alpha = 0.6
            ax.plot(i, j, linewidth=linewidth, alpha=alpha)

        # mean_tpr = np.mean(tps, axis=0)
        # mean_fpr = np.linspace(0, 1, 100)
        # std_tpr = np.std(tps, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.axis('tight')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        # ax.grid(False)

        roc_auc = np.array(roc_auc)
        ax.text(0.8, 0.05,
                'AUC = {:2.2f} +- {:2.2f}'.format(roc_auc.mean(), roc_auc.std()),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes)

    @staticmethod
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


class StatisticalTester:

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def do_t_test(self, res_dir, reference_file=None):
        assert os.path.exists(res_dir), "specified directory not found."
        scores_paths = [os.path.join(res_dir, p) for p in os.listdir(res_dir) if
                        p.endswith('.npy') and (not p.startswith('train_test_indices'))]
        if not scores_paths:
            print('Can not find any score file.')
            return
        if reference_file is not None:
            l = [i for i in scores_paths if i.split('/')[-1] == reference_file]
            if not l:
                print('reference file not found.')
                return
            comb = [(l[0], i) for i in scores_paths if i != l[0]]
            for res1_path, res2_path in comb:
                self._ttest(res2_path, res1_path)
                self._ttest(res1_path, res2_path)
        else:
            comb = combinations(scores_paths, 2)
            for res1_path, res2_path in comb:
                self._ttest(res1_path, res2_path)
                self._ttest(res2_path, res1_path)

    def _ttest(self, res1_path, res2_path):
        """Does a less-than test, i.e. tests the null hypothesis of res1_path's
         measure is equal or less than the res2's, versus the alternative hypothesis
         of "res1_path's measure is higher than res2_path's".
        """
        fn1 = res1_path.split('/')[-1]
        fn2 = res2_path.split('/')[-1]
        print("H0: x({}) <= x({})".format(fn1, fn2))
        acc_diff, fscore_diff, l_diff = self._get_diffs_mode1(res1_path, res2_path)

        t_stat, p_val = ttest_1samp(acc_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' Accuracies:')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

        t_stat, p_val = ttest_1samp(fscore_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' F1-scores:')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

        t_stat, p_val = ttest_1samp(l_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' Losses:')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

        acc_diff, fscore_diff = self._get_diffs_mode2(res1_path, res2_path)

        t_stat, p_val = ttest_1samp(acc_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' Accuracies (SW):')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

        t_stat, p_val = ttest_1samp(fscore_diff, 0)
        rejection = (p_val / 2 < self.alpha) and (t_stat > 0)
        print(' F1-scores (SW):')
        print('     Rejection: ', rejection)
        if rejection:
            print('     P-value: ', p_val)

    @staticmethod
    def _get_diffs_mode1(res1_path, res2_path):
        res1 = np.load(res1_path, allow_pickle=True)[:, 0]
        res2 = np.load(res2_path, allow_pickle=True)[:, 0]

        l_diff = np.zeros(100)
        acc_diff = np.zeros(100)
        fscore_diff = np.zeros(100)
        for i in range(100):
            l1, acc1, fscore_1 = res1[i][:3]
            l2, acc2, fscore_2 = res2[i][:3]
            l_diff[i] = l1 - l2
            acc_diff[i] = acc1 - acc2
            fscore_diff[i] = fscore_1 - fscore_2
        return acc_diff, fscore_diff, l_diff

    def _get_diffs_mode2(self, res1_path, res2_path):
        res1 = np.load(res1_path, allow_pickle=True)[:, 1]
        res2 = np.load(res2_path, allow_pickle=True)[:, 1]

        acc_diff = np.zeros(100)
        fscore_diff = np.zeros(100)
        for i in range(100):
            acc1, fscore1 = self._get_subject_wise_scores(res1[i])
            acc2, fscore2 = self._get_subject_wise_scores(res2[i])

            acc_diff[i] = acc1 - acc2
            fscore_diff[i] = fscore1 - fscore2
        return acc_diff, fscore_diff

    @staticmethod
    def _get_subject_wise_scores(res):
        tns, fps, fns, tps = res
        acc_vector = (tps + tns) / (tps + fns + fps + tns)
        prec_vector = tps / (tps + fps + 0.0001)
        rec_vector = tps / (tps + fns + 0.0001)
        spec_vector = tns / (tns + fps + 0.0001)
        fscore_vector = 2 * (prec_vector * rec_vector) / (prec_vector + rec_vector + 0.0001)
        return acc_vector, fscore_vector
