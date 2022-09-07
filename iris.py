import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    data = set_data("./data/problema_iris.csv")
    regList = regression(data)
    graph(data, regList)

def regression(data):
    # Se calcula el ajuste lineal para cada par de variables
    regList = []
    reg = LinearRegression()
    for i in range(1,5):
        for j in range(1,5):
            lr = []
            if i == j:
                continue
            else:
                # Se asignan los datos de las variables a relacionar
                x = data[:,i]
                y = data[:,j]
                # fit
                reg.fit(x.reshape(-1, 1), y)
                m = reg.coef_
                b = reg.intercept_
                message = "Datos\nCoficiente: {}, Intercepto: {}".format(m, b) 
                print(message)
                # Ajuste lineal
                xp = np.linspace(min(x), max(x), num=2)
                yp = m*xp+b
                # R2
                r2 = round(reg.score(x.reshape(-1, 1), y), 3)
                # Se almacenan los datos de cada relacion
                lr.append(xp)
                lr.append(yp)
                lr.append(r2)
            regList.append(lr)
    return regList

def set_data(url):
    # Lee los datos desde archivo csv
    data = np.asarray(np.genfromtxt(url, delimiter=","))
    return data

def graph(data, regList):
    # Subplots para mostrar varias graficas simultaneamente
    figure, axis = plt.subplots(4, 3)
    figure.suptitle("RELACION DE VARIABLES EN PROBLEMA DE IRIS.")

    axis[0, 0].plot(data[:,1], data[:,2], "b.", label="Datos")
    axis[0, 0].plot(regList[0][0], regList[0][1], "r-", label="Ajuste Lineal")
    axis[0, 0].legend(["SL & SW. (cm)", "R2: {}".format(regList[0][2])])
    axis[0, 0].grid()

    axis[0, 1].plot(data[:,1], data[:,3], "b.", label="Datos")
    axis[0, 1].plot(regList[1][0], regList[1][1], "r-", label="Ajuste Lineal")
    axis[0, 1].legend(["SL & PL. (cm)", "R2: {}".format(regList[1][2])])
    axis[0, 1].grid()
    
    axis[0, 2].plot(data[:,1], data[:,4], "b.", label="Datos")
    axis[0, 2].plot(regList[2][0], regList[2][1], "r-", label="Ajuste Lineal")
    axis[0, 2].legend(["SL & PW. (cm)", "R2: {}".format(regList[2][2])])
    axis[0, 2].grid()
    
    axis[1, 0].plot(data[:,2], data[:,1], "b.", label="Datos")
    axis[1, 0].plot(regList[3][0], regList[3][1], "r-", label="Ajuste Lineal")
    axis[1, 0].legend(["SW & SL. (cm)", "R2: {}".format(regList[3][2])])
    axis[1, 0].grid()
    
    axis[1, 1].plot(data[:,2], data[:,3], "b.", label="Datos")
    axis[1, 1].plot(regList[4][0], regList[4][1], "r-", label="Ajuste Lineal")
    axis[1, 1].legend(["SW & PL. (cm)", "R2: {}".format(regList[4][2])])
    axis[1, 1].grid()
    
    axis[1, 2].plot(data[:,2], data[:,4], "b.", label="Datos")
    axis[1, 2].plot(regList[5][0], regList[5][1], "r-", label="Ajuste Lineal")
    axis[1, 2].legend(["SW & PW. (cm)", "R2: {}".format(regList[5][2])])
    axis[1, 2].grid()
    
    axis[2, 0].plot(data[:,3], data[:,1], "b.", label="Datos")
    axis[2, 0].plot(regList[6][0], regList[6][1], "r-", label="Ajuste Lineal")
    axis[2, 0].legend(["PL & SL. (cm)", "R2: {}".format(regList[6][2])])
    axis[2, 0].grid()
    
    axis[2, 1].plot(data[:,3], data[:,2], "b.", label="Datos")
    axis[2, 1].plot(regList[7][0], regList[7][1], "r-", label="Ajuste Lineal")
    axis[2, 1].legend(["PL & SW. (cm)", "R2: {}".format(regList[7][2])])
    axis[2, 1].grid()
    
    axis[2, 2].plot(data[:,3], data[:,4], "b.", label="Datos")
    axis[2, 2].plot(regList[8][0], regList[8][1], "r-", label="Ajuste Lineal")
    axis[2, 2].legend(["PL & PW. (cm)", "R2: {}".format(regList[8][2])])
    axis[2, 2].grid()
    
    axis[3, 0].plot(data[:,4], data[:,1], "b.", label="Datos")
    axis[3, 0].plot(regList[9][0], regList[9][1], "r-", label="Ajuste Lineal")
    axis[3, 0].legend(["PW & SL. (cm)", "R2: {}".format(regList[9][2])])
    axis[3, 0].grid()
    
    axis[3, 1].plot(data[:,4], data[:,2], "b.", label="Datos")
    axis[3, 1].plot(regList[10][0], regList[10][1], "r-", label="Ajuste Lineal")
    axis[3, 1].legend(["PW & SW. (cm)", "R2: {}".format(regList[10][2])])
    axis[3, 1].grid()
    
    axis[3, 2].plot(data[:,4], data[:,3], "b.", label="Datos")
    axis[3, 2].plot(regList[11][0], regList[11][1], "r-", label="Ajuste Lineal")
    axis[3, 2].legend(["PW & PL. (cm)", "R2: {}".format(regList[11][2])])
    axis[3, 2].grid()
    
    plt.show()



if __name__ == "__main__":
    main()
