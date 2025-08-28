import numpy as np
from numpy import ndarray
from typing import Callable
import math

def deriv(func: Callable[[ndarray], ndarray],
          _input: ndarray, 
          delta: float=0.001) -> ndarray:
    return (func(_input + delta) - func(_input - delta))/ (delta * 2)

def cube(x: ndarray) -> ndarray:
    return x ** 3
def square(x: ndarray) -> ndarray:
    return x ** 2
def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)
def sigmoid(x: ndarray)-> ndarray:
    return (1 / (1 + np.exp(-x)))