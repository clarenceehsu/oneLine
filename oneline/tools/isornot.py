
def isPrime(number):
    if number % 2 == 0:
        return False
    for i in range(3, int(number ** 0.5) + 1, 2):
        if number % i == 0:
            return False
    return True

def isOdd(number):
    return number % 2

def isEven(number):
    return not number % 2
