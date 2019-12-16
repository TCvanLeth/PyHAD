#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:01:17 2017

@author: tcvanleth
"""
import numpy as np

def curve_fit(func, x, y, p0, sigma, jac):
    p = p0
    lmpar = lmpar0
    while True:
        J = jac(x, p)
        res = y - func(x, p)
        JW = J.T @ weight
        JWJ = JW @ J
        delta = np.linalg.solve(JWJ + lmpar * np.diag(JWJ), JW * res)
        p = p + delta
        if np.isclose(delta, 0):
            break
    return p