from os import system

def F1(shop):
    system("cls")
    a = input("Введите название товара: ")
    if a in shop:
        print(shop[a][0])
    else:
        print("Товар не найден.")
    input()

def F2(shop):
    system("cls")
    a = input("Введите название товара: ")
    if a in shop:
        print(shop[a][1])
    else:
        print("Товар не найден.")
    input()

def F3(shop):
    system("cls")
    a = input("Введите название товара: ")
    if a in shop:
        print(shop[a][2])
    else:
        print("Товар не найден.")
    input()

def F4(shop):
    system("cls")
    for i in shop:
        print(f"{i} - {shop[i][0]}, {shop[i][1]}, {shop[i][2]}")
    input()

def F5(shop):
    system("cls")
    a = input("Введите название товара: ")
    b = int(input("Введите количество: "))
    if a in shop:
        if b <= shop[a][2]:
            print(f"Стоимость: {shop[a][1] * b}")
            print(f"Остаток: {shop[a][2] - b}")
            shop[a][2] -= b
        else:
            print("Недостаточно товара.")
    else:
        print("Товар не найден.")
    input()



shop = {
    "Кольца": ["Золото", 1000, 5],
    "Серьги": ["Серебро", 750, 7],
    "Браслеты": ["Золото-серебро", 950, 11],
    "Ожерелья": ["Драгоценные камни", 1500, 4],
    "Законки": ["Золото", 350, 18]
}

F = {
    1: F1,
    2: F2,
    3: F3,
    4: F4,
    5: F5,
}

print(f"Количество функций: {len(F)}")



while True:
    system("cls")
    print("1.Просмотр описания")
    print("2.Просмотр цены")
    print("3.просмотр количества")
    print("4.Вся информация")
    print("5.Покупка")
    print("6.Выход")
    a = int(input())
    if a == 6:
        break
    if a in F:
        F[a](shop)
