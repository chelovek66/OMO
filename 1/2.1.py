from sympy import primerange

def Prime_number(a):
    if a < 10:
        return F(a,30)
    elif a < 100:
        return F(a,550)
    elif a < 1000:
        return F(a,10000)

def F(a, b):
    L = list(primerange(0,b))
    return L[a-1]

print("Введи номер простого числа")
a = int(input())
print(Prime_number(a))