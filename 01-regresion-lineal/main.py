import numpy as np
import pandas as pd

from regresiones.regresion_clase import RegresionLinealSimple

#Leyendo el dataset
publicidad = pd.read_csv('Advertising.csv',index_col = 0)
tv = publicidad['Newspaper']


sales = publicidad['Sales']

#Pasamos datos a arrays
x = np.array(tv)
y = np.array(sales)

print(x)


rl = RegresionLinealSimple(x,y)
b1 = rl.beta1
b0 = rl.beta0

print("pendiente: ", b1)
print("interceptor: ", b0)

print("Y = {} + {}X".format(b0,b1))

rl.plot_recta()


#Hacemos una prediccion
prediccion = b0 + b1*215;
print(prediccion)