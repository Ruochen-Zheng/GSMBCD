def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
      
  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
'''
def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
      try:
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]
      except:
        fixed_blocks[i] = block_indices[i*block_size:]

    return fixed_blocks
'''

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = []
    for i in range(n_blocks):
      fixed_blocks.append(block_indices[i*block_size: i*block_size + block_size])
    return np.array(fixed_blocks)import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
      
  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
'''
def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
      try:
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]
      except:
        fixed_blocks[i] = block_indices[i*block_size:]

    return fixed_blocks
'''

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = []
    for i in range(n_blocks):
      fixed_blocks.append(block_indices[i*block_size: i*block_size + block_size])
    return np.array(fixed_blocks)import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
      
  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
'''
def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
      try:
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]
      except:
        fixed_blocks[i] = block_indices[i*block_size:]

    return fixed_blocks
'''

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = []
    for i in range(n_blocks):
      fixed_blocks.append(block_indices[i*block_size: i*block_size + block_size])
    return np.array(fixed_blocks)import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
      
  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
'''
def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
      try:
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]
      except:
        fixed_blocks[i] = block_indices[i*block_size:]

    return fixed_blocks
'''

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = []
    for i in range(n_blocks):
      fixed_blocks.append(block_indices[i*block_size: i*block_size + block_size])
    return np.array(fixed_blocks)import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
      
  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
'''
def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
      try:
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]
      except:
        fixed_blocks[i] = block_indices[i*block_size:]

    return fixed_blocks
'''

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = []
    for i in range(n_blocks):
      fixed_blocks.append(block_indices[i*block_size: i*block_size + block_size])
    return np.array(fixed_blocks)import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
      
  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
'''
def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
      try:
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]
      except:
        fixed_blocks[i] = block_indices[i*block_size:]

    return fixed_blocks
'''

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = []
    for i in range(n_blocks):
      fixed_blocks.append(block_indices[i*block_size: i*block_size + block_size])
    return np.array(fixed_blocks)import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
      
  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
'''
def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
      try:
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]
      except:
        fixed_blocks[i] = block_indices[i*block_size:]

    return fixed_blocks
'''

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = []
    for i in range(n_blocks):
      fixed_blocks.append(block_indices[i*block_size: i*block_size + block_size])
    return np.array(fixed_blocks)import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
 # if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
 #   assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB" :
      return None
      
  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      #n_blocks = int(n_params / block_size)
      n_blocks = int(round(n_params / block_size))
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
      
  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)


  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  #np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks
'''
def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
      try:
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]
      except:
        fixed_blocks[i] = block_indices[i*block_size:]

    return fixed_blocks
'''

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = []
    for i in range(n_blocks):
      fixed_blocks.append(block_indices[i*block_size: i*block_size + block_size])
    return np.array(fixed_blocks)
