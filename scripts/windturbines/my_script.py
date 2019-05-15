# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:08:36 2019

@author: jschmidt
"""

import imp
import scripts.windturbines.my_module
imp.reload(scripts.windturbines.my_module)

import scripts.windturbines.my_module as m
imp.reload(m)
