import warnings
from collections import Counter
from numbers import Integral

import numpy as np

from sklearn.neighbors._base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
import six
import joblib
from sklearn.utils.multiclass import type_of_target

SAMPLING_KIND = ('over-sampling', 'under-sampling', 'clean-sampling',
                 'ensemble')
TARGET_KIND = ('binary', 'multiclass', 'multilabel-indicator')


def check_neighbors_object(nn_name, nn_object, additional_neighbor=0):
    if isinstance(nn_object, Integral):
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    elif isinstance(nn_object, KNeighborsMixin):
        return nn_object


def check_target_type(y, indicate_one_vs_all=False):
    type_y = type_of_target(y)
    if type_y not in TARGET_KIND:
        # FIXME: perfectly we should raise an error but the sklearn API does
        # not allow for it
        warnings.warn("'y' should be of types {} only. Got {} instead.".format(
            TARGET_KIND, type_of_target(y)))

    if indicate_one_vs_all:
        return (y.argmax(axis=1) if type_y == 'multilabel-indicator' else y,
                type_y == 'multilabel-indicator')
    else:
        return y.argmax(axis=1) if type_y == 'multilabel-indicator' else y


def hash_X_y(X, y, n_samples=10, n_features=5):
    row_idx = slice(None, None, max(1, X.shape[0] // n_samples))
    col_idx = slice(None, None, max(1, X.shape[1] // n_features))

    return joblib.hash(X[row_idx, col_idx]), joblib.hash(y[row_idx])


def _ratio_all(y, sampling_type):
    """Returns ratio by targeting all classes."""
    target_stats = Counter(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        ratio = {key: n_sample_majority - value
                 for (key, value) in target_stats.items()}
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        n_sample_minority = min(target_stats.values())
        ratio = {key: n_sample_minority for key in target_stats.keys()}
    else:
        raise NotImplementedError

    return ratio


def _ratio_majority(y, sampling_type):
    """Returns ratio by targeting the majority class only."""
    if sampling_type == 'over-sampling':
        raise ValueError("'ratio'='majority' cannot be used with"
                         " over-sampler.")
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        target_stats = Counter(y)
        class_majority = max(target_stats, key=target_stats.get)
        n_sample_minority = min(target_stats.values())
        ratio = {key: n_sample_minority
                 for key in target_stats.keys()
                 if key == class_majority}
    else:
        raise NotImplementedError

    return ratio


def _ratio_not_minority(y, sampling_type):
    """Returns ratio by targeting all classes but not the minority."""
    target_stats = Counter(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        ratio = {key: n_sample_majority - value
                 for (key, value) in target_stats.items()
                 if key != class_minority}
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        ratio = {key: n_sample_minority
                 for key in target_stats.keys()
                 if key != class_minority}
    else:
        raise NotImplementedError

    return ratio


def _ratio_minority(y, sampling_type):
    """Returns ratio by targeting the minority class only."""
    target_stats = Counter(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        ratio = {key: n_sample_majority - value
                 for (key, value) in target_stats.items()
                 if key == class_minority}
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        raise ValueError("'ratio'='minority' cannot be used with"
                         " under-sampler and clean-sampler.")
    else:
        raise NotImplementedError

    return ratio


def _ratio_auto(y, sampling_type):
    """Returns ratio auto for over-sampling and not-minority for
    under-sampling."""
    if sampling_type == 'over-sampling':
        return _ratio_all(y, sampling_type)
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        return _ratio_not_minority(y, sampling_type)


def _ratio_dict(ratio, y, sampling_type):
    """Returns ratio by converting the dictionary depending of the sampling."""
    target_stats = Counter(y)
    # check that all keys in ratio are also in y
    set_diff_ratio_target = set(ratio.keys()) - set(target_stats.keys())
    if len(set_diff_ratio_target) > 0:
        raise ValueError("The {} target class is/are not present in the"
                         " data.".format(set_diff_ratio_target))
    # check that there is no negative number
    if any(n_samples < 0 for n_samples in ratio.values()):
        raise ValueError("The number of samples in a class cannot be negative."
                         "'ratio' contains some negative value: {}".format(
            ratio))
    ratio_ = {}
    if sampling_type == 'over-sampling':
        n_samples_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        for class_sample, n_samples in ratio.items():
            if n_samples < target_stats[class_sample]:
                raise ValueError("With over-sampling methods, the number"
                                 " of samples in a class should be greater"
                                 " or equal to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                    target_stats[class_sample], n_samples))
            if n_samples > n_samples_majority:
                warnings.warn("After over-sampling, the number of samples ({})"
                              " in class {} will be larger than the number of"
                              " samples in the majority class (class #{} ->"
                              " {})".format(n_samples, class_sample,
                                            class_majority,
                                            n_samples_majority))
            ratio_[class_sample] = n_samples - target_stats[class_sample]
    elif sampling_type == 'under-sampling':
        for class_sample, n_samples in ratio.items():
            if n_samples > target_stats[class_sample]:
                raise ValueError("With under-sampling methods, the number of"
                                 " samples in a class should be less or equal"
                                 " to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                    target_stats[class_sample], n_samples))
            ratio_[class_sample] = n_samples
    elif sampling_type == 'clean-sampling':
        # clean-sampling can be more permissive since those samplers do not
        # use samples
        for class_sample, n_samples in ratio.items():
            ratio_[class_sample] = n_samples
    else:
        raise NotImplementedError

    return ratio_


def check_ratio(ratio, y, sampling_type, **kwargs):
    if sampling_type not in SAMPLING_KIND:
        raise ValueError("'sampling_type' should be one of {}. Got '{}'"
                         " instead.".format(SAMPLING_KIND, sampling_type))

    if np.unique(y).size <= 1:
        raise ValueError("The target 'y' needs to have more than 1 class."
                         " Got {} class instead".format(np.unique(y).size))

    if sampling_type == 'ensemble':
        return ratio

    if isinstance(ratio, six.string_types):
        if ratio not in RATIO_KIND.keys():
            raise ValueError("When 'ratio' is a string, it needs to be one of"
                             " {}. Got '{}' instead.".format(RATIO_KIND,
                                                             ratio))
        return RATIO_KIND[ratio](y, sampling_type)
    elif isinstance(ratio, dict):
        return _ratio_dict(ratio, y, sampling_type)
    elif callable(ratio):
        ratio_ = ratio(y, **kwargs)
        return _ratio_dict(ratio_, y, sampling_type)


RATIO_KIND = {'minority': _ratio_minority,
              'majority': _ratio_majority,
              'not minority': _ratio_not_minority,
              'all': _ratio_all,
              'auto': _ratio_auto}
