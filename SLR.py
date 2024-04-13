import pandas as pd
from sklearn.linear_model import LinearRegression


class SimpleLinearRegression:
    class CAL:
        def __init__(self):
            self.address = 'Book2023_CAL.csv'
            self.inclination = []
            self.intercept = []

    class IOW:
        def __init__(self):
            self.address = 'Book2023_IOW.csv'
            self.inclination = []
            self.intercept = []

    class NEB:
        def __init__(self):
            self.address = 'Book2023_NEB.csv'
            self.inclination = []
            self.intercept = []

    class VIR:
        def __init__(self):
            self.address = 'Book2023_VIR.csv'
            self.inclination = []
            self.intercept = []

    def __init__(self):
        cal = self.CAL()
        iow = self.IOW()
        neb = self.NEB()
        vir = self.VIR()

        self.state = {'CA': cal, 'IA': iow, 'NE': neb, 'VA': vir}
        state = (cal, iow, neb, vir)

        for i in range(len(state)):
            df = pd.read_csv(state[i].address)
            df.head()

            year = df[['year']]
            gdp = df[['GDP']]
            pop_u20_m = df[['poplation(20-)(male)']]
            pop_o20_m = df[['poplation(20+)(male)']]
            pop_o65_m = df[['poplation(65+)(male)']]
            pop_u20_f = df[['poplation(20-)(female)']]
            pop_o20_f = df[['poplation(20+)(female)']]
            pop_o65_f = df[['poplation(65+)(female)']]
            emp_u20_f = df[['employment(20-)(female)']]
            emp_o20_f = df[['employment(20+)(female)']]
            emp_o65_f = df[['employment(65+)(female)']]
            emp_u20_m = df[['employment(20-)(male)']]
            emp_o20_m = df[['employment(20+)(male)']]
            emp_o65_m = df[['employment(65+)(male)']]

            element = (gdp, pop_u20_m, pop_o20_m, pop_o65_m, pop_u20_f, pop_o20_f, pop_o65_f,
                       emp_u20_f, emp_o20_f, emp_o65_f, emp_u20_m, emp_o20_m, emp_o65_m)

            for j in range(len(element)):
                model_lr = LinearRegression()
                model_lr.fit(year, element[j])

                state[i].inclination.append(model_lr.coef_)
                state[i].intercept.append(model_lr.intercept_)

    def submit_prediction(self, state, year):
        result = []
        data = self.state[state]

        for i in range(13):
            cal = year * data.inclination[i] + data.intercept[i]
            result.append(cal[0][0])

        return result
