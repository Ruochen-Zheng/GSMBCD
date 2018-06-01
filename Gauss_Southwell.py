from __future__ import division
from . import tree_algorithms as ta
import numpy as np
from update_rules import update_rules as ur
import copy
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from itertools import cycle

########### ------------------------------ ADAPTIVE RULES
def select(rule, x,sig,A, b, loss, args, iteration,x_outer,miu):
    """ Adaptive selection rules """
    n_params = x.size
    block_size = args["block_size"]
    it = iteration
    lipschitz = loss.lipschitz
    #partial computes
    pvb=0
      
    g_func = loss.g_func
       
    if rule in ["GS"]:
      """ select coordinates based on largest gradients"""
      #g = g_func(x, A, b, block=None)
  
      if sig=='None-revised':

        g=miu

      elif sig=='revised':
        
        g=g_func(x, A, b, block=None)-g_func(x_outer, A, b, block=None)+miu
        pvb=2*A.shape[0]

      elif sig=='3':
        g=g_func(x, A, b, block=None)
        pvb=A.shape[0]

      s = np.abs(g+np.sign(x)*args["L1"])
      #print max(s)
      block = np.argsort(s, axis=None)[-block_size:]
      #print block
##
