#ЛР1 З6 В4

from random import randint

K = tuple(randint(1,10) for _ in range(10))

print(*K)
print(max(K))
print(min(K))