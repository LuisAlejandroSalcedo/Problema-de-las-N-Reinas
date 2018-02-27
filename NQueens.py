# Problema de las N reinas. El problema consiste en colocar N reinas en
# un tablero de dimensiones N x N, de forma que el número de ataques entre
# reinas seal el menor posible
#
# Simplificaciones: Se considera que sólo hay una reina por fila y columna
# por lo que los ataques sólo puedes ser diagonales  
# 
# Individuos: Los individuos son una lista en la que el índice es la columna
# donde se colola la reina y el contenido es la fila.
#
#############################################################################

import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

#Parámetros del problema 
NB_QUEENS = 50

# FUNCIÓN DE FITNESS
def evalNQueens(individual):
    """ Función de evaluación del problema. El problema consiste en posiciones
    N reinas en un tablero de ajedrez de dimensiones NxN. La función de evaluación
    calcula el número de reinas R que hay en cada diagonal. El número de ataques
    que hay en cada diagonal se puede calcular como R-1. 
    """
    size = len(individual)
    # Los ataques sólo pueden ser en las diagonales
    diagonal_izquierda_derecha = [0] * (2*size-1)
    diagonal_derecha_izquierda = [0] * (2*size-1)
    
    # Número de reinas en cada diagonal
    for i in range(size): # recorremos las columnas
        diagonal_izquierda_derecha[i+individual[i]] += 1 # [columna + fila]
        diagonal_derecha_izquierda[size-1-i+individual[i]] += 1 # [size-1-columna+ fila]
    
    # Número de ataques en cada diagonal
    suma = 0
    for i in range(2*size-1): # recorremos todas las diagonales
        if diagonal_izquierda_derecha[i] > 1: # hay ataques
            suma += diagonal_izquierda_derecha[i] - 1 # n-1 ataques
        if diagonal_derecha_izquierda[i] > 1:
            suma += diagonal_derecha_izquierda[i] - 1
    return suma,


# DEFINICIÓN DEL PROBLEMA
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# REGISTRO DE FUNCIONES QUE SON NECESARIAS -- CAJA DE HERRAMIENTAS
toolbox = base.Toolbox()
toolbox.register("permutation", random.sample, range(NB_QUEENS), NB_QUEENS)

# Funciones de inicilización del individuo y de la población
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de evaluación
toolbox.register("evaluate", evalNQueens)

# Operadores genéticos
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0/NB_QUEENS)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    seed=0
    random.seed(seed)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1) # objeto que almacena el mejor individuo
    stats = tools.Statistics(lambda ind: ind.fitness.values) # objeto para calcular estadísticas
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                        halloffame=hof, verbose=True) # algoritmo genético como "caja negra"

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, best = main()
    print(best)
    print(best[0].fitness.values)
    y = best[0]
    x= range(NB_QUEENS)
    x= numpy.array(x)
    print(x)
    y = numpy.array(y)
    print(y)    
    x = x + 0.5
    y = y + 0.5
    plt.figure()
    plt.scatter(x,y)
    plt.xlim(0,NB_QUEENS)
    plt.ylim(0,NB_QUEENS)
    plt.xticks(x-0.5)
    plt.yticks(x-0.5)
    plt.grid(True)
    plt.title(u"Mejor Individuo")
    plt.show()