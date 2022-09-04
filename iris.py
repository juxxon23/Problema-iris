import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
    # Crear cada relacion de forma automatica
    x, y = set_data("./data/problema_iris.csv", 1, 2)
    xp, yp = regression(x, y)
    graph(x, y, xp, yp)

def regression(x, y):
    reg = LinearRegression()
    reg.fit(x.reshape(-1, 1), y)
    m = reg.coef_
    b = reg.intercept_
    message = "Datos\nCoficiente: {}, Intercepto: {}".format(m, b) 
    print(message)
    xp = np.linspace(min(x), max(x), num=2)
    yp = m*xp+b
    return xp, yp

def set_data(url, xvar, yvar):
    data = np.asarray(np.genfromtxt(url, delimiter=","))
    x = data[:, xvar]
    y = data[:, yvar]
    return x, y

def graph(x, y, xp, yp):
    figure, axis = plt.subplots(2, 3)
    figure.suptitle("Relacion de variables en problema de Iris")

    axis[0, 0].plot(x, y, "b.", label="Datos")
    axis[0, 0].plot(xp, yp, "r-", label="Ajuste Lineal")
    axis[0, 0].set_title("SL & SW. (cm)")
    axis[0, 0].legend
    axis[0, 0].grid()

    axis[0, 0].plot(x, y, "b.", label="Datos")
    axis[0, 0].plot(xp, yp, "r-", label="Ajuste Lineal")
    axis[0, 0].set_title("SL & SW. (cm)")
    axis[0, 0].legend
    axis[0, 0].grid()
    
    plt.show()



if __name__ == "__main__":
    main()
