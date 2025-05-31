#Diamond game

# Estás jugando un juego en el que un comerciante tiene un lote de bolsas a la venta. 
# Cada bolsa contiene un número desconocido de diamantes. 
# Puedes comprar una bolsa seleccionada al azar por 12 monedas, mirar dentro de la bolsa 
# para ver cuántos diamantes tiene, y decidir inmediatamente si quieres 
# quedártela o venderla de vuelta por 11 monedas. 
# También puedes, tantas veces como quieras y sin costo, echar un vistazo 
# a una bolsa aleatoria y contar los diamantes, pero sin la opción de comprarla. 
# ¿Cuál sería una buena estrategia para maximizar la cantidad de diamantes en las 
# bolsas que decides quedarte?

import functools
import matplotlib.pyplot as plt
import random
from typing import *
from statistics import mean, stdev
from collections import Counter
from typing import Union, Iterable, Mapping
from statistics import mean, stdev


BUY = 12
SELL = 11
MEAN = 100
BAGS = 1000
PEEKS = 100

cache = functools.lru_cache(None)
random.seed(42)


# Esta clase permite que los objetos sean hashables (es decir, que se puedan usar como claves en diccionarios o sets).
class Hashable:
    def __hash__(self):
        # Devuelve un hash único basado en la identidad del objeto
        return id(self)

    def __eq__(self, other):
        # Compara si dos objetos son exactamente el mismo (no solo si tienen los mismos valores)
        return self is other



# Esta clase representa una distribución de probabilidad discreta
class ProbDist(Hashable, Counter):
    """Una distribución de probabilidad discreta, que mapea valores a frecuencias."""

    def __init__(self, samples: Union[Iterable, Mapping], name='ProbDist'):
        """
        Inicializa la distribución.
        `samples` puede ser una lista de valores (iterable) o un diccionario {valor: frecuencia}.
        `name` es un nombre opcional para la distribución.
        """
        # Inicializa Counter con los valores (cuenta cuántas veces aparece cada uno)
        super().__init__(samples)

        # Obtiene una lista de todos los elementos, respetando las frecuencias
        values = list(self.elements())

        # Calcula la media (μ) y la desviación estándar (σ) de los valores
        self.μ = mean(values)
        self.σ = stdev(values)
        self.n = len(values)  # Número total de muestras
        self.name = name      # Nombre de la distribución

    def __call__(self, d) -> float:
        # Devuelve la probabilidad de un valor d (frecuencia relativa)
        return self[d] / self.n

    def __str__(self) -> str:
        # Representación amigable en texto de la distribución
        return f'{self.name}[μ={self.μ:.1f}, σ={self.σ:.1f}, n={self.n:,d}]'



# Creamos una distribución de probabilidad con 5 muestras (algunos valores repetidos)
P = ProbDist([108, 92, 108, 100, 92])

# Verificamos que el número 108 aparece 2 veces
assert P[108]  == 2   # número de veces que aparece 108

# Verificamos que la probabilidad de obtener 108 es 2/5 (porque aparece 2 veces de un total de 5)
assert P(108)  == 2/5 # probabilidad de que salga 108

# Verificamos que el total de muestras es 5
assert P.n     == 5   # número total de muestras en la distribución

# Verificamos que la media (promedio) de los valores es 100
assert P.μ     == 100 # media de las muestras

# Verificamos que la desviación estándar es 8
assert P.σ     == 8   # desviación estándar de las muestras

# Verificamos que el diccionario interno almacena correctamente las frecuencias
assert dict(P) == {108: 2, 92: 2, 100: 1} # conteo de cada valor

# Verificamos que la representación técnica (útil para depuración) es correcta
assert repr(P) == 'ProbDist({108: 2, 92: 2, 100: 1})' # representación interna

# Verificamos que la representación en texto (str) muestra resumen de la distribución
assert str(P)  == 'ProbDist[μ=100.0, σ=8.0, n=5]' # resumen amigable con media, desviación y tamaño


# Voy a definir la función `normal` para crear un conjunto de bolsas que tendrá 
# una distribución aproximadamente normal (gaussiana). Luego, 
# definiré la función `peek` para echar un vistazo a algunas de las bolsas en ese 
# conjunto.


def normal(σ, μ=MEAN, n=BAGS) -> ProbDist:
    """
    Crea una distribución de probabilidad aproximadamente normal (gaussiana).
    
    Parámetros:
    - σ: desviación estándar (indica qué tan dispersos están los valores).
    - μ: media (valor promedio esperado). Por defecto usa la constante MEAN.
    - n: número de bolsas o muestras a generar. Por defecto usa la constante BAGS.
    
    Retorna:
    - Una instancia de ProbDist con los valores generados.
    """
    # Creamos una lista de 'n' valores generados con una distribución normal (gaussiana)
    # Cada valor se redondea y se asegura que no sea menor que 0.
    values = [max(0, round(random.gauss(μ, σ))) for _ in range(n)]
    
    # Devolvemos una distribución de probabilidad con estos valores, nombrándola "normal"
    return ProbDist(values, "normal")


def peek(stockpile: ProbDist, peeks=PEEKS) -> ProbDist:
    """
    Simula echar un vistazo a algunas bolsas del stockpile (muestra parcial aleatoria).
    
    Parámetros:
    - stockpile: una distribución de probabilidad existente (el conjunto total de bolsas).
    - peeks: cuántas bolsas observar al azar. Por defecto se usa la constante PEEKS.
    
    Retorna:
    - Una nueva distribución de probabilidad basada en las bolsas observadas.
    """
    # Tomamos 'peeks' bolsas aleatoriamente del stockpile, considerando su frecuencia (peso)
    values = random.choices(list(stockpile), list(stockpile.values()), k=peeks)
    
    # Devolvemos una nueva distribución con los valores observados, nombrándola como "_peek"
    return ProbDist(values, stockpile.name + "_peek")


# Creamos un stockpile de bolsas con una desviación estándar de 1
stock1 = normal(σ=1)

# Echamos un vistazo a 100 bolsas al azar del stockpile
peek1 = peek(stock1, peeks=100)

# Mostramos en pantalla la información estadística del stockpile completo
print("Mostramos en pantalla la información estadística del stockpile completo : ____", stock1)  
# Salida esperada: normal[μ=100.0, σ=1.1, n=1,000]
# Esto indica que el promedio de diamantes es aproximadamente 100, con poca dispersión (σ ≈ 1.1)

# Mostramos en pantalla la información estadística de la muestra ("peek")
print( "Mostramos en pantalla la información estadística de la muestra PEEK ___" , peek1)   
# Salida esperada: normal_peek[μ=100.0, σ=1.0, n=100]
# A pesar de haber tomado solo 100 bolsas, la muestra representa muy bien la distribución original


# aqui imprimimos las mas comunes resultados de cada distribucion 
print("mas comunes del total de bolsas", stock1.most_common()) # Most common outcomes and their counts
print("mas comune de las muestras", peek1.most_common())



# Estrategias
# Explicamos anteriormente que lo mejor es hacer todas las observaciones (peeks) primero, y 
# luego continuar comprando bolsas hasta que se agoten las monedas. Por lo tanto, asumiremos 
# que todas las estrategias hacen eso, y nos queda la tarea de tomar decisiones de guardar/vender. 
# Utilizaré dos convenciones:

# Una estrategia es una función strategy(c, d) que devuelve 'keep' (guardar) si un jugador que 
# acaba de comprar una bolsa con d diamantes y le quedan c monedas debería quedarse con la bolsa; 
# de lo contrario, devuelve 'sell' (vender).

# Un generador de estrategias es una función a la que se le pasa una distribución de bolsas observadas 
# (peeked-at) y devuelve una función de estrategia.

# Por ejemplo, cutoff_strategy es un generador de estrategias tal que cutoff_strategy(peeks, 0.95) 
# devuelve una función de estrategia que guarda todas las bolsas que tienen al menos el 95% del número 
# promedio de diamantes en las bolsas observadas (peeks). (Además, si el jugador comenzó exactamente con 
# la cantidad de monedas necesarias para comprar una bolsa, la guarda, porque venderla no le daría suficientes 
# monedas para comprar otra).

from typing import Callable, Literal

# Definimos los tipos:
Action = Literal["keep", "sell"]  # Una acción solo puede ser "keep" (quedarse con la bolsa) o "sell" (venderla)
Strategy = Callable[[int, int], Action]  # Una estrategia es una función que toma monedas y diamantes y devuelve una acción

# Función que genera una estrategia basada en un corte (cutoff)
def cutoff_strategy(peek: ProbDist, ratio=1.0) -> Strategy:
    """
    Genera una estrategia que decide quedarse con una bolsa solo si tiene
    al menos una cierta proporción del promedio de diamantes observados.

    Parámetros:
    - peek: distribución de diamantes obtenida al observar varias bolsas.
    - ratio: proporción del promedio que se usará como límite mínimo para quedarse con una bolsa.

    Retorna:
    - Una función (estrategia) que toma la cantidad de monedas restantes (c)
      y la cantidad de diamantes en la bolsa (d), y decide si quedarse o venderla.
    """

    # Calculamos el punto de corte mínimo de diamantes para quedarse con la bolsa
    cutoff = ratio * peek.μ

    def strategy(c: int, d: int) -> Action:
        """
        Decide si quedarse con la bolsa o venderla.

        Parámetros:
        - c: monedas que le quedan al jugador
        - d: diamantes en la bolsa actual

        Regla:
        - Si el jugador tiene justo la cantidad inicial de monedas para comprar (BUY),
          se queda con la bolsa (porque venderla no permitiría comprar otra).
        - Si la bolsa tiene diamantes >= al corte, se la queda.
        - En otro caso, la vende.
        """
        return "keep" if (c == BUY) or d >= cutoff else "sell"

    # Damos un nombre descriptivo a la estrategia (útil para depuración o impresión)
    strategy.__name__ = f"cutoff_{cutoff:.0f}"

    return strategy



# Creamos una estrategia basada en los datos observados en peek1.
# Esta estrategia mantendrá las bolsas que tengan al menos el 95% del promedio de diamantes en peek1.
strategy = cutoff_strategy(peek1, 0.95)

# Probamos la estrategia con un jugador que tiene 42 monedas y una bolsa con 92 diamantes.
# Si 92 está por debajo del umbral (95% del promedio), la estrategia debe indicar "sell" (vender).
assert strategy(42, 92) == "sell"

# Probamos con otra bolsa que tiene 103 diamantes y el mismo número de monedas.
# Si 103 es igual o mayor al umbral, la estrategia debe indicar "keep" (quedarse con la bolsa).
assert strategy(42, 103) == "keep"




# Voy a definir
# como el número esperado de diamantes que se pueden obtener con
# monedas, a partir de una reserva con la distribución
# , cuando el jugador sigue la estrategia dada
# .

# El valor esperado es 0 si no tenemos suficientes monedas para comprar ninguna bolsa, y 
# en caso contrario, es el promedio ponderado por probabilidad, sobre todas las bolsas posibles, 
# del valor esperado del resultado de quedarse con la bolsa o venderla de vuelta, dependiendo de lo 
# que indique la estrategia.

# Cuando el costo de una bolsa es 12, esto se puede escribir como:

# (fórmula no incluida aquí)

# Nuestro juego es lo suficientemente pequeño como para que sea computacionalmente factible 
# calcular exactamente el valor esperado, siempre y cuando almacenemos en caché los resultados.


from functools import cache

@cache  # Decorador que guarda (cachea) los resultados para no repetir cálculos costosos
def E(P: ProbDist, strategy: Strategy, c: int) -> float:
    """
    Calcula el número esperado de diamantes que se pueden obtener con `c` monedas,
    usando una estrategia `strategy`, y asumiendo que las bolsas tienen una distribución `P`.
    
    Parámetros:
    - P: distribución de probabilidad de diamantes por bolsa (ProbDist)
    - strategy: función que decide si quedarse con una bolsa o venderla
    - c: número de monedas disponibles

    Devuelve:
    - Un valor flotante que representa el número esperado de diamantes obtenidos.
    """

    # Si no tenemos suficientes monedas para comprar una bolsa, el resultado esperado es 0
    if c < BUY:
        return 0

    # Si tenemos monedas suficientes, evaluamos todas las posibles bolsas d en P
    # Para cada cantidad de diamantes d, usamos su probabilidad P(d)
    # y calculamos cuánto diamantes obtendríamos según la estrategia:
    # - Si la estrategia dice "keep", ganamos d diamantes y gastamos BUY monedas
    # - Si dice "sell", no ganamos diamantes pero recuperamos SELL monedas
    # Luego sumamos todas las posibilidades ponderadas por sus probabilidades
    return sum(
        P(d) * (
            E(P, strategy, c - BUY) + d  # caso "keep": obtenemos d diamantes y perdemos BUY monedas
            if strategy(c, d) == 'keep'
            else E(P, strategy, c - BUY + SELL)  # caso "sell": recuperamos SELL monedas
        )
        for d in P  # iteramos sobre todas las posibles cantidades de diamantes en la distribución
    )


# Esta función siempre termina porque cada llamada recursiva reduce la cantidad de monedas, y cuando 
# llegamos a tener menos de 12 monedas, el valor esperado siempre es cero. Si este no fuera el caso 
# (por ejemplo, si se pudiera tener un saldo negativo de monedas y aún así continuar el juego), 
# entonces este cálculo del valor esperado llevaría a un bucle infinito.


# Estrategia óptima
# Para cualquier juego, la estrategia óptima consiste en tomar la acción que conduzca al mayor valor esperado. 
# Eso puede sonar trivial o circular: “la mejor estrategia es elegir la mejor acción”. 
# Pero en realidad no es trivial ni circular, y puede implementarse usando un generador de estrategias llamado optimal_strategy.


def optimal_strategy (peek: ProbDist) -> Strategy:
    # Strategua oara tmar la  accion (mantener o vender) que lleve al mejor alo esperado
    def optimal (c: int, d: int) -> Action:
        return "keep" if E(peek, optimal, c - BUY) + d > E(peek, optimal, c - BUY + SELL) else "sell"
    return optimal


# Un punto sutil pero importante: si queremos conocer el valor esperado real de una situación, simplemente llamamos a 
# E(c, stockpile, strategy). Eso es bastante sencillo para nosotros como observadores externos del juego.
# Pero un jugador dentro del juego no tiene acceso al stockpile (la reserva completa de bolsas); 
# todo lo que tiene es una estimación de la distribución del stockpile, obtenida a partir de las observaciones (peeks). 
# Veremos que, por lo general, esta estimación es bastante buena.

stock10 = normal(σ=10)
peek10 = peek(stock10)
optimal = optimal_strategy(peek10)

# aqui validamos que los resultads serian coretcos 
assert 99 < peek10.μ < 101
assert 9 < peek10.σ < 11
assert optimal(13,98) == "sell"
assert optimal(13,103) == "keep"
assert optimal(15,103) == "sell"
assert optimal(23,110) == "sell"
assert optimal(23,116) == "keep"
assert optimal(12,50) == "keep"


# Simulamos el juego usando una estrategia dada, comenzando con cierta cantidad de monedas
# y una distribución de bolsas con diamantes. La función devuelve las bolsas que el jugador decide conservar.



def play(strategy, coins: int, stockpile: ProbDist, verbose=True) -> List[int]:
    """
    Ejecuta una partida usando una `strategy` (estrategia), y devuelve la lista de bolsas que el jugador decidió conservar.
    Si `verbose=True`, también se imprimen las decisiones que se van tomando durante el juego.
    """
    # Convertimos la distribución de probabilidad en una lista real de bolsas para poder elegir aleatoriamente.
    bags = list(stockpile.elements())

    # Esta lista guardará los valores (cantidad de diamantes) de las bolsas que el jugador decide conservar.
    kept = []

    # Mientras tengamos monedas suficientes para comprar una bolsa (el costo está definido por la constante BUY):
    while coins >= BUY:
        coins -= BUY  # Gastamos monedas para comprar una bolsa
        bag = random.choice(bags)  # Elegimos una bolsa aleatoria del stockpile
        action = strategy(coins, bag)  # Aplicamos la estrategia para decidir si conservar o vender la bolsa

        if action == 'keep':
            kept.append(bag)  # Si decidimos conservarla, la añadimos a la lista de bolsas guardadas
        else:
            coins += SELL  # Si decidimos venderla, recuperamos algunas monedas (definidas por la constante SELL)

        # Si verbose es True, imprimimos lo que ocurrió en esta ronda
        if verbose:
            print(f'{bag:3d} diamond bag: {action} it (total: {sum(kept):3d} diamonds and {coins:3d} coins.')

    # Cuando ya no nos alcanzan las monedas para seguir comprando, terminamos y devolvemos las bolsas guardadas
    return kept

# Ejecutamos el juego usando la estrategia óptima basada en un conjunto de 10 bolsas observadas
play(optimal_strategy(peek10), 95, stock10)

# Ahora guardamos el resultado en una variable para poder hacer algo más con ella después
kept_bags = play(optimal_strategy(peek10), 95, stock10)

# Imprimimos las bolsas que el jugador decidió conservar (cada número es la cantidad de diamantes de esa bolsa)
print("Bolsas mantenidas:", kept_bags)


# Visualizando una Estrategia
# Hemos definido la estrategia óptima, pero... ¿realmente la entendemos?
# ¿En qué situaciones exactas deberíamos quedarnos con una bolsa?
# ¿Cuándo conviene venderla?
# ¿Podemos dar una explicación clara e intuitiva del porqué?
# A veces, ver una estrategia en acción ayuda a entenderla mejor. Para eso, podemos hacer una visualización que muestre, para cada combinación de monedas disponibles (c) y diamantes en la bolsa (d), si la estrategia decide “keep” o “sell”.
# El código que viene a continuación sirve para eso: crear una tabla o gráfico que muestre cuándo se guarda o se vende una bolsa, en función del número de monedas que te quedan y de cuántos diamantes trae la bolsa que acabas de comprar.

plt.rcParams["figure.figsize"] = (14, 6)

COINS = range(12, 121) # Range of coin values to examine

def plot_strategy(P: ProbDist, strategy):
    """Plot (coins, diamonds) points for which strategy(c, d) == 'sell'."""
    points = [(c, d) for c in COINS for d in range(min(P), max(P) + 1) 
              if strategy(c, d) == 'sell']
    plt.scatter(*transpose(points), marker='.', label=strategy.__name__)
    decorate(title=f'When to sell back with {strategy.__name__} strategy on {P}')
    
def decorate(title, xticks=COINS[::BUY], xlabel='coins', ylabel='diamonds'):
    """Decorate the plot with title, grid lines, ticks, and labels."""
    plt.grid(True); plt.xticks(xticks)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    
def transpose(matrix): return zip(*matrix)


plot_strategy(stock10, optimal_strategy(stock10))
plt.show()


# Interpretando la Gráfica de la Estrategia Óptima
# El gráfico a continuación analiza situaciones en las que:
# El eje X muestra la cantidad de monedas que tienes antes de comprar una bolsa.
# El eje Y muestra la cantidad de diamantes en la bolsa que compraste.
# Un punto azul en la posición (x, y) significa que la estrategia óptima decide vender la bolsa 
# en esa situación específica.
# Esto representa la verdadera estrategia óptima para el conjunto de datos stock10, obtenida 
# mediante un cálculo exacto de probabilidades. Es importante notar que:
# ⚠️ Esta información exacta no está disponible para un jugador real, ya que requiere 
# conocimiento total del contenido del stock (todas las bolsas posibles y sus probabilidades), 
# algo que normalmente no se conoce con certeza en el juego.