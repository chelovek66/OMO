from random import randint

n = 5
m = 5
List = []
Sum = []
for i in range(n):
    List_ = []
    for j in range(m):
        List_.append(randint(1,10))
    List.append(List_)

print(*List,sep='\n')
    
for i in range(n):
    Sum.append(sum(List[i]))

print(Sum)