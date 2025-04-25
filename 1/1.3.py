import math

def is_perfect_square(x):
    if x < 0:
        return False
    root = math.isqrt(x)
    return root * root == x

def solve_for_k(k):

    C = (k * (k - 1) * (2 * k - 1)) // 6
    
    n = 1
    while n <= k//2+1: 
        S_k = k * n**2 + n * (k**2 - k) + C
        
        if S_k > 0 and is_perfect_square(S_k):
            return n
        n += 1

    return None

t = int(input())  
ks = [int(input()) for _ in range(t)]

for k in ks:
    n = solve_for_k(k)
    if n:
        print("Yes")
        print(n)
    else:
        print("No")
