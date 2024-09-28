import math as m

def addition(*args):
    """
    This function is for adding to numbers in python
    """
    res = 0
    for a in args:
        if isinstance(a, int):
            res += a
    return res

def multiplication(*nums):
    """
    This functions is for multiplication of numbers, you can pass 'n' number of parameter
    """
    res = 1
    for num in nums: res *= num
    return res

def square_num(x): 
    """
    This functions is forr squaring given number x
    """
    return x**2

def power_num(x, y):
    """
    This functions is for calculting a power of 'x' to 'y'
    """
    return m.pow(x, y)

def square_root(x):
    """
    This functions is for square root if a number 'x'
    """
    return m.sqrt(x)