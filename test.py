# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:37:55 2019

@author: jschmidt
"""
import ml_classes
reload(ml_classes)
linear=Function_data_lstm(n,10,max_elem,model,linear_f,1)
linear.fit_model_plot_results(epochs,batch_size)