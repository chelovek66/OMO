import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH = r'C:\\Users\\user\\OneDrive\\Рабочий стол\\ОМО ЛР\\2\\1.csv'

t = pd.read_csv(PATH)


t['Income'] = t['Income'].astype(str).str.replace(r'[$,\s]', '', regex=True)
t['Income'] = t['Income'].astype(float)

t.info()
print(t.describe())

t['TotalSpending'] = t[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
t['Rest'] = t['Income']-t['TotalSpending']
t['IncomeLevel'] = pd.cut(t['Income'],bins=[0, 30000, 70000, float('inf')],labels=['Small', 'Medium', 'High'])

print(t)

sns.set_style("whitegrid")
plt.subplot(2,2,1)
sns.histplot(t['Income'], bins=30, kde=True)
plt.title("Распределение доходов")
plt.xlabel("Доход")
plt.ylabel("Частота")

plt.subplot(2,2,2)
sns.scatterplot(x=t['Income'], y=t['TotalSpending'])
plt.title("Связь между доходом и расходами")
plt.xlabel("Доход")
plt.ylabel("Общие расходы")

plt.subplot(2,2,3)
sns.countplot(x=t['IncomeLevel'])
plt.title("Распределение клиентов по уровню дохода")
plt.xlabel("Уровень дохода")
plt.ylabel("Количество клиентов")

plt.subplot(2,2,4)
sns.boxplot(x=t['Income'])
plt.title("Boxplot доходов")
plt.xlabel("Доход")
plt.show()

class DataPipeLine:
        def __init__(self):
                self.TotalSpending_m = None
                self.Rest_m = None
                self.IncomeLevel_m = None


        def F(self,t):
                self.TotalSpending_m = t['TotalSpending'].median()
                self.Rest_m = t['Rest'].median()
                self.IncomeLevel_m = 'Medium'

        @staticmethod
        def NoNull_I(t):
                for col in t.columns:
                        if t[col].isnull().any():
                                if t[col].dtype in ['float64', 'int64']:
                                        mid = t[col].median()
                                        t[col] = t[col].fillna(mid)
                                else:
                                        mid = t[col].mode()[0]
                                        t[col] = t[col].fillna(mid)

        @staticmethod

        def NoNull_O(t):
            return t[t['Year_Birth'] >= 1935]
                
        
DP = DataPipeLine()
DP.NoNull_I(t)
DP.F(t)
t = DP.NoNull_O(t)

t.info()

sns.histplot(t['Year_Birth'], bins=30, kde=True)
plt.title("Распределение доходов")
plt.xlabel("Доход")
plt.ylabel("Частота")
plt.show()