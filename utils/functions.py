#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helping functions for 'introduction' and 'quickstart' notebooks."""

__authors__ = 'Xinyi Liu, Wanxia Deng, Lei Liu, Jinfeng Ding, Xiao Cheng'
__copyright__ = ['National University of Defense Technology School of Meteorology and Oceanology']
__contact__ = ['liuxinyi20@nudt.edu.cn', 'dengwanxia14@nudt.edu.cn']
__version__ = '1.0.0'
__date__ = '2026-04-30'

# -- Built-in modules -- #

# -- Third-party modules -- #
from sklearn.metrics import r2_score


def r2_metric(true, pred):
    """
    Calculate the r2 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.

    Returns
    -------
    r2 : float
        The calculated r2 score.

    """
    r2 = r2_score(y_true=true, y_pred=pred)

    return {
        "R2 Score": r2
    }