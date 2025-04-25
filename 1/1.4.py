#ЛР1 З4 В4

from random import randint

N = int(input())

L1 = [i for i in range(1,N + 1)]
L2 = [randint(1,N) for _ in range(1,N + 1)]

print(L1)
print(L2, "\n")

Dict = {}

for i in range(N):
    Dict[L1[i]] = 0

for i in L2:
    Dict[i] += 1

print(Dict)
# print(L1)
# print(L2)