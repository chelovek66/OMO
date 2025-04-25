

class dog:

    def __init__(self):
        self.Dog = ["Bulldog", "Beagle"]

    def P(self):
        print(self.Dog)

class cat:

    def __init__(self):
        self.Cat = ["Persian", "Siamese"]

    def P(self):
        print(self.Cat)


class animal(cat,dog):
    a = "Статическое свойство"
    def __init__(self):
        super().__init__()
        dog.__init__(self)  
        cat.__init__(self) 
        self.Animal = [self.Dog , self.Cat]

    def P(self):    
        print(self.Animal)

    def P1(cls):
        print(cls.a)

dog().P()
cat().P()
animal().P()
animal().P1()