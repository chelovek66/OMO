

class Stationery:

    def __init__(self, title):
        self.title = title

    def draw(self):
        print(f"Запуск отрисовки: {self.title}")

class Pen(Stationery):

    def __init__(self, title, N):
        super().__init__(title)
        self.N = N

    def draw(self):
        super().draw()
        print(f"Количество кучек - {self.N}")

class Pencil(Stationery):

    def __init__(self, title, N):
        super().__init__(title)
        self.N = N

    def draw(self):
        super().draw()
        print(f"Количество карандашей - {self.N}")

class Handle(Stationery):

    def __init__(self, title, N):
        super().__init__(title)
        self.N = N

    def draw(self):
        super().draw()
        print(f"Количество маркеров - {self.N}")


s = Stationery("Канцелярия")
P = Pen("Ручка", 10)
p = Pencil("Карандаш", 5)
H = Handle("Маркер", 1)


s.draw()
P.draw()
p.draw()
H.draw()