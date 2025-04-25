a = 10
b = 0

try:
    r = a / b
except ZeroDivisionError:
    print("Ошибка")
    r = None
finally:
    print("+")

print(f"Результат деления: {r}")
