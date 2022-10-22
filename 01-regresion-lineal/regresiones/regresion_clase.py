from numpy import *
import numpy as np
import matplotlib.pyplot as plt


class RegresionLinealSimple:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.beta1 = self.pendiente(x,y)
        self.beta0 = self.interceptor(x,y)

    def pendiente(self,x,y):
        termino1 = x-average(x)
        termino2 = y-average(y)
        Sxy = sum(termino1*termino2)
        Sxx = sum(termino1*termino1)

        return Sxy/Sxx

    def interceptor(self,x,y):
        return average(y)-self.beta1*average(x)


    def plot_recta(self):
        b1 = self.beta1
        b0 = self.beta0

        #Para dinujar el modelo de RegresionLineal(La linea)
        puntos_x = linspace(np.amin(self.x),np.amax(self.x),10)
        puntos_y = b0+b1*puntos_x

        print("Valor de b1: ", b1)
        print("Valor de b0: ", b0)

        #Para dibujar los puntos del dataset
        plt.plot(puntos_x, puntos_y)
        plt.plot(self.x,self.y,'r*')
        plt.show()

