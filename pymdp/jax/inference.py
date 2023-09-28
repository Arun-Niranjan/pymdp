#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

from .algos import run_factorized_fpi, run_mmp, run_vmp
from jax import tree_util as jtu

def update_posterior_states(A, B, obs, past_actions, prior=None, qs_hist=None, A_dependencies=None, num_iter=16, method='fpi'):

    if method == 'fpi' or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else:
        # format B matrices using action sequences here
        B = jtu.tree_map(lambda b, a_idx: b[..., a_idx].transpose(3, 0, 1, 2), B, past_actions) # assumes there is a batch dimension

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, num_iter=num_iter) 
        if method == 'mmp':
            qs = run_mmp(A, B, obs, prior, A_dependencies, num_iter=num_iter)

    return qs_hist.append(qs)
    
