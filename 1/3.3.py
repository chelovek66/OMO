
Faculty = ["Факультет информатики", "Факультет математики", "Факультет экономики", "Факультет физики", "Факультет юриспруденции"]
FIO = ["Иванов Иван Иванович", "Петров Петр Петрович", "Сидорова Анна Сергеевна", "Кузнецов Алексей Владимирович", "Новикова Ольга Александровна"]
Age = [20, 21, 22, 19, 20]
Results = [[5, 4, 3], [4, 5, 5], [3, 4, 4], [5, 5, 5], [4, 3, 4]]

class Faculty:

    def __init__(self, faculty):
        self.faculty = faculty

    def set_faculty(self, faculty):
        self.faculty = faculty

class Student:

    def __init__(self,FIO,Age,Results):
        self.FIO = FIO
        self.age = Age
        self.results = Results

    def set_FIO(self, FIO):
        self.FIO = FIO

    def set_Age(self, Age):
        self.age = Age

    def set_Results(self, Results):
        self.results = Results

    def Output(self, name):
        if name in self.FIO:
            idx = self.FIO.index(name)
            S = sum(self.results[idx]) / len(self.results[idx])
            print(f"Сердний балл студента: {S}")

St = Student(FIO,Age,Results)

while True:
    print("Введите ФИО: ") 
    name = input()
    St.Output(name)

    