"""
This is a simple content that contains many easy-to-use and efficient functions for judgement.
"""


def is_prime(number):
    """
    Return True if the input number is a prime number.
    :param number: the number required
    :return: bool
    """
    if number % 2 == 0:
        return False
    for i in range(3, int(number ** 0.5) + 1, 2):
        if number % i == 0:
            return False
    return True


def is_odd(number):
    """
    Return True if the input number is a odd number.
    :param number: the number required.
    :return: bool
    """
    return bool(number % 2)


def is_even(number):
    """
    Return True if the input number is an even number.
    :param number: the number required.
    :return: bool
    """
    return not number % 2
