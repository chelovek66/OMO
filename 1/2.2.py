



# arg = { 1 : 2 , 2 : 1 , 3 : 5 } 
# arg = [0,1,2,3,4,5,6,0,7,8,9,10,0,0]
# arg = 28
arg = "шалаш"



def F(arg):
    T = type(arg)
    print()
    if T == dict:
        Min_v = min(arg.values())
        for K, V in arg.items():
            if V == Min_v:
                print(K,V)
                break
    elif T == list:
        R = 1
        coun_0 = arg.count(0)
        # print(coun_0)
        if arg.count(0) in [0,1]:
            print(0)
        else:
            for i in range(arg.index(0)+1,arg[arg.index(0)+1::].index(0)):
                # print(arg[i])
                R *= i
            print(R)
        new_arg = list(set(arg))
        print(new_arg)
    elif T == int:
        for i in range(1,arg+1):
            if arg % i == 0:
                print(i)
    elif T == str:
        vowels = 'аеёиоуыэюя'
        consonants = 'бвгджзйклмнпрстфхцчшщ'

        v = 0
        c = 0

        if arg == arg[::-1]:
            print("Является полиндромом")
        
        for i in arg:
            if i in vowels:
                v += 1
            elif i in consonants:
                c +=1

        print(f"Гласных - {v}, согласных - {c}")


F(arg)