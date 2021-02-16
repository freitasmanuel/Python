from numpy import random

'''
    Optimizador Harmony Search Algorithm (HSA). Python - Ing. Manuel Freitas 2020.
    
    Ejemplo:

            Maximizar F(x) = 6.5 * X1 + 7 * X2

            Restricciones:

                X1      > 0
                X2      > 0
                x + y   < 500
                2x + 3y < 600
                2x + y  < 400
    
    Resultado: 
                F(x) = 1675

                X1 = 150, X2 = 100

'''

# No Variables
No_Var = 2

# No Individuos
No_Ind = 50

# No de Iteraciones
max_iter = 50000

# Límites de las Variables
low = 0.0
high = 1000.0

# Parámetros del HSA

HCMR = 0.0
HCMR_Min = 0.95
HCMR_Max = 0.98

Par = 0.0
Par_Min = 0.95
Par_Max = 0.98

Bw = 0.0
Bw_Min = 0.01
Bw_Max = (high - low) / 20

def Fx(x, iter):

    valor = 6.5 * x[0] + 7 * x[1]

    # Restricciones
    g = np.array(np.zeros(5), dtype = float)
    
    # X1 > 0
    g[0] = min(0, x[0]) ** 2
    
    # X2 > 0
    g[1] = min(0, x[1]) ** 2
    
    # x + y < 500
    g[4] = max(0, (x[0] + x[1]) - 500) ** 2

    # 2x + 3y < 600
    g[2] = max(0, (2 * x[0] + 3 * x[1]) - 600) ** 2
    
    # 2x + y < 400
    g[3] = max(0, (2 * x[0] + x[1]) - 400) ** 2

    restric = 0.0

    for i in range(len(g)):
        restric = g[i]  * iter * 10 + restric 

    valor = valor - restric

    return  valor

def HSA(f, x_start):

    '''
        @param f (funcion): función a optimizar
        @param x_start (numpy arreglo): Arreglo con valores iniciales
        return: tuple (arreglo con el mejor score, Mejor score)
    '''

    # Inicio

    prev_best = f(x_start, 1)
    res = [[x_start, prev_best]]

    for i in range(No_Ind-1):
        
        x = x_start.copy()

        for j in range(No_Var):
            x[j] = random.uniform(low, high, 1)

        score = f(x,1)

        res.append([x, score])

    # Iteraciones

    iters = 0
    ant = -1e150

    while 1:

        # Ordenar por F(x) decreciente
        res.sort(reverse=True, key=lambda x: x[1])
        
        best = res[0][1]

        if best > ant:
            print("F(x): %10.2f | " % (best), end='')
            
            for i in range(No_Var):
                print("X[%d] = %10.2f | " % (i+1, res[0][0][i]), end='')

            print(" Iter.: ", iters, "/", max_iter)

            ant = best

        # Salir al alcanzar el máximo de las iteraciones
        if max_iter and iters >= max_iter:
            return res[0]

        iters += 1

        HCMR = HCMR_Min + ( (HCMR_Max - HCMR_Min) / max_iter) * iters 
        Par = Par_Min + ( (Par_Max - Par_Min) / max_iter) * iters 
        Bw = Bw_Max * math.exp((math.log(Bw_Min/Bw_Max)/max_iter) * iters)

        x = np.array(np.zeros(No_Var), dtype = float)
        
        for i in range(No_Var):

            if random.uniform(0.0, 1.0, 1) <= HCMR:
            
                pos = int(random.uniform(0.0, No_Ind, 1))

                y = res[pos][0].copy()

                x[i] = y[i]

                if random.uniform(0.0, 1.0, 1) <= Par:
                    if random.uniform(0.0, 1.0, 1) < 0.5:
                        x[i] = x[i] - Bw * random.uniform(0.0, 1.0, 1)
                    else:
                        x[i] = x[i] + Bw * random.uniform(0.0, 1.0, 1)

            else:

                x[i] = random.uniform(low, high, 1)
                
            if x[i] < low:
                x[i] = low
            else:
                if x[i] > high:
                    x[i] = high

        rscore = f(x, iters)

        # Si el score de x es mejor al peor score de la población, 
        # se reemplaza el peor individuo por x

        if rscore >= res[No_Ind-1][1]:
            res[No_Ind-1][0] = x.copy()
            res[No_Ind-1][1] = rscore

if __name__ == "__main__":

    import math
    import numpy as np

    # Se inicializa el primer individuo de la población
    x = np.array(np.zeros(No_Var), dtype = float)

    for i in range(No_Var):
        x[i] = random.uniform(low, high, 1)

    # Algoritmo HSA

    resp = HSA(Fx, x)

    # Imprimir el resultado

    print("")
    print("   Máximo F(x): %10.2f" % (resp[1]))
    print("   Mejor Solución: ", end='')

    print(" | ", end='')

    for i in range(No_Var):
        print("X[%d] = %5.2f | " % (i+1, resp[0][i]), end='')

    print(" ")

    print("")