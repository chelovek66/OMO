
def solve_for_k(k):
    
    for n in range(1,10**18):

        S_k = k*n**2+n*(k**2 - k)+((k-1)*k*(2*k-1))//6     
        if S_k <= 0 :
            return None
        rez = int(S_k**0.5)
        rez = rez ** 2
        if rez  == S_k:
            return n
        elif rez > S_k:
            return None

# Read input
t = int(input())
ks = [int(input()) for i in range(t)]


for k in ks:
    n = solve_for_k(k)
    if n is not None:
        print("Yes")
        print(n)
    else:
        print("No")
