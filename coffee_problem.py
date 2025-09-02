#!/usr/bin/env pyton3
'''
Solve the coffee problem to learn how to drink coffee effectively.
'''

import numpy as np


def solve_temp(Tenv=20., T0=90., k=1/300., t):
    '''
    This function returns temperature as a function of time using Newton's Law of Cooling.
    ---------
    '''
    
    T_coffee = Tenv + (T0-Tenv)*np.exp(-k*t)

