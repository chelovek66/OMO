
# def Del(R):
#     a = [1]

#     for i in range(2,int(R**0.5)+1):
#         if R % i == 0:
#             a.append(i)
#             if i != R // i:
#                 a.append(R//i)
#     a.append(R)
#     return a

# nums = input()
# nums = nums.replace('*','')
# nums = list(nums)

# rez = 1
# ABC = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}

# AB = {}

# for i in reversed(nums):
#     if i != '=':
#         rez *= int(i)
#     else:
#         break

# for i in nums:
#     if i not in ABC and i != '=':
#         rez /= int(i)
#     elif i in ABC:
#         if i not in AB:
#             AB[i] = 1
#         else:
#             AB[i] += 1
#     elif i == '=':
#         rez = int(rez)
#         break

# AB =sorted(AB.values(), reverse=True)

# n = len(AB)

# S = 0

# if n == 1:
#     print(rez)
# else:
#     LD = Del(rez)
#     LD = sorted(LD)
#     LD_set = set(LD)
#     for i in range(n):
        

# print(nums,end='\n')
# print(rez,end='\n')
# print(LD,end='\n')
# print(len(LD))
# print(AB)

def Del(R):

    a = [1]
    for i in range(2, int(R**0.5) + 1):
        if R % i == 0:
            a.append(i)
            if i != R // i:
                a.append(R // i)
    a.append(R)
    return sorted(a)

def factorial(n):

    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def count_unique_permutations(degrees):

    total = sum(degrees)
    numerator = factorial(total)
    denominator = 1
    for degree in degrees:
        denominator *= factorial(degree)
    return numerator // denominator

def extract_degrees(expression):

    variables = {}
    for term in expression.split('*'):
        if term in variables:
            variables[term] += 1
        elif term.isalpha(): 
            variables[term] = 1
    return list(variables.values())

def generate_groups(divisors, target, degrees):

    def helper(current_group, remaining_divisors, current_product, index, usage_count):
        if len(current_group) == len(degrees):
            if current_product == target:
                groups.append(current_group)
            return
        if index >= len(remaining_divisors):
            return

        if usage_count[remaining_divisors[index]] < degrees[len(current_group)]:
            new_group = current_group + [remaining_divisors[index]]
            new_product = current_product * remaining_divisors[index]
            if new_product <= target:
                usage_count[remaining_divisors[index]] += 1
                helper(new_group, remaining_divisors, new_product, index, usage_count)
                usage_count[remaining_divisors[index]] -= 1

        helper(current_group, remaining_divisors, current_product, index + 1, usage_count)

    groups = []
    usage_count = {d: 0 for d in divisors}  
    helper([], divisors, 1, 0, usage_count)
    return groups


MOD = 10**9 + 7

nums = input("")


degrees = extract_degrees(nums.split('=')[0])


nums = nums.replace('*', '')
left, right = nums.split('=')
rez = 1


for char in right:
    if char.isdigit():
        rez *= int(char)


for char in left:
    if char.isdigit() and int(char) != 0:
        rez //= int(char)

if rez == 0:
    print("-1")
else:

    divisors = Del(rez)

    successful_groups = generate_groups(divisors, rez, degrees)

    permutations_count = count_unique_permutations(degrees)

    total_groups = len(successful_groups)
    result = (total_groups * permutations_count) % MOD
    print(result)