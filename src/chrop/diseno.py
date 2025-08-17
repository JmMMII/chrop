import numpy as np
import sympy as sp
import bisect as bs
import itertools
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class DiseNo:
    """
    Representa un diseño experimental con puntos y pesos asociados.

    Métodos:
        - copia(): Devuelve una copia del diseño.
        - multiplicar_p(num): Multiplica todos los pesos por un escalar.
        - aNadir(x, p): Añade un nuevo punto al diseño, ajustando los pesos existentes.
        - quitar(i): Elimina el punto en la posición i.
        - refinar(m, cercania, pmin): Refina el diseño fusionando puntos cercanos y eliminando puntos con poco peso.
    """
    class Punto:
        """
        Representa un punto de diseño con coordenada x y peso p.
        """
        def __init__(self, x : float, p : float):
            """
            Inicializa un punto con coordenada x y peso p.

            Args:
                x (float): Coordenada del punto.
                p (float): Peso asociado al punto.
            """
            self.x = x
            self.p = p

        def __lt__(self, punto):
            """
            Permite comparar dos puntos por su coordenada x.

            Args:
                punto (Punto): Otro punto a comparar.

            Returns:
                bool: True si self.x < punto.x, False en caso contrario.
            """
            return self.x < punto.x
        
        def __str__(self):
            """
            Representación en cadena del punto.

            Returns:
                str: Cadena con formato "x : p".
            """
            return f"{self.x} : {self.p}"

    def __init__(self, puntos : list[tuple[float,float]]):
        """
        Inicializa el diseño con una lista de puntos (x, p).

        Args:
            puntos (list[tuple[float, float]]): Lista de tuplas (coordenada, peso).
        """
        self.puntos = list()
        for x,p in puntos:
            self.puntos.append(self.Punto(x,p))

    def copia(self):
        """
        Devuelve una copia profunda del diseño.

        Returns:
            DiseNo: Nuevo objeto DiseNo con los mismos puntos.
        """
        return DiseNo([(punto.x, punto.p) for punto in self.puntos])

    def __str__(self):
        """
        Representación en cadena del diseño.

        Returns:
            str: Cadena con todos los puntos y pesos.
        """
        return "\n".join(f"{punto.x} : {punto.p}" for punto in self.puntos)

    def multiplicar_p(self, num : float):
        """
        Multiplica todos los pesos de los puntos por un escalar.

        Args:
            num (float): Escalar por el que se multiplican los pesos.
        """
        for punto in self.puntos:
            punto.p *= num

    def aNadir(self, x : float, p : float):
        """
        Añade un nuevo punto al diseño y ajusta los pesos existentes.

        Args:
            x (float): Coordenada del nuevo punto.
            p (float): Peso del nuevo punto.
        """
        self.multiplicar_p(1-p)
        self.puntos.append(self.Punto(x,p))

    def quitar(self, i : int):
        """
        Elimina el punto en la posición i.

        Args:
            i (int): Índice del punto a eliminar.
        """
        self.puntos.pop(i)

    #Ver que valores le asigno a cada uno
    def refinar(self, m : int, cercania : float = 0.001, pmin : float = 0.001):
        """
        Refina el diseño fusionando puntos cercanos y eliminando puntos con peso bajo.
        Garantiza al menos m puntos en el diseño.

        Args:
            m (int): Número mínimo de puntos requeridos.
            cercania (float, optional): Distancia máxima para fusionar puntos. Default 0.001.
            pmin (float, optional): Peso mínimo para conservar un punto. Default 0.001.

        Raises:
            ValueError: Si no hay suficientes puntos tras el refinamiento.
            ValueError: Si el peso total retirado es mayor o igual a 1.
        """
        # Fusionar puntos muy cercanos por coordenada x
        self.puntos.sort(key=lambda p: p.x)
        i = 0
        while i < len(self.puntos) - 1:
            if abs(self.puntos[i].x - self.puntos[i+1].x) <= cercania:
                peso_total = self.puntos[i].p + self.puntos[i+1].p
                x_fusion = (self.puntos[i].x * self.puntos[i].p + self.puntos[i+1].x * self.puntos[i+1].p) / peso_total
                self.puntos[i].x = x_fusion
                self.puntos[i].p = peso_total
                self.puntos.pop(i+1)
            else:
                i += 1

        # Separar puntos por peso usando búsqueda binaria (lista ordenada por peso)
        puntos_por_p = sorted(self.puntos, key=lambda punto: punto.p)  # ascendente por p
        pesos = [p.p for p in puntos_por_p]
        idx = bs.bisect_left(pesos, pmin)

        quitar = puntos_por_p[:idx]    # objetos Punto con p < pmin
        buenos = puntos_por_p[idx:]    # objetos Punto con p >= pmin

        # Si hay menos de m puntos en buenos, restaurar los de mayor peso desde 'quitar'
        if len(buenos) < m:
            quitar.sort(key=lambda p: p.p, reverse=True)
            while len(buenos) < m and quitar:
                buenos.append(quitar.pop(0))
            if len(buenos) < m:
                raise ValueError("No hay suficientes puntos para completar el diseño.")

        # suma de los pesos de los puntos que se quitaron
        s = sum(p.p for p in quitar)

        denom = 1.0 - s
        if denom <= 0:
            raise ValueError("Peso total retirado >= 1, no se puede normalizar")
        for punto in buenos:
            punto.p /= denom

        self.puntos = sorted(buenos, key=lambda punto: punto.x)

# Función que separa los términos del modelo
def terminos(modelo : str, var : sp.Symbol) -> list[sp.Expr]:
    """
    Separa los términos simbólicos de un modelo en función de la variable dada.

    Args:
        modelo (str): Expresión simbólica del modelo (por ejemplo, "x**2 + x").
        var (sp.Symbol): Variable simbólica sobre la que se separan los términos.

    Returns:
        sp.Matrix: Vector columna con los términos simbólicos que contienen la variable.
    """
    # Expandir y convertir a suma de términos
    modelo = sp.expand(modelo)
    terminos = modelo.as_ordered_terms()

    # Filtrar solo los que contienen la variable
    terminos_con_var = [str(t) for t in terminos if t.has(var)]

    return sp.Matrix(sp.sympify(terminos_con_var))

# Función que calcula el valor de los terminos del modelo sustituyendo la variable por valor
def funcion(f_vector: sp.Matrix, variable: sp.Symbol, valor: float) -> sp.Matrix:
    """
    Evalúa el vector de términos simbólicos sustituyendo la variable por un valor numérico.

    Args:
        f_vector (sp.Matrix): Vector columna de términos simbólicos.
        variable (sp.Symbol): Variable simbólica a sustituir.
        valor (float): Valor numérico para la variable.

    Returns:
        sp.Matrix: Vector columna evaluado numéricamente.
    """
    sustituciones = {variable: valor, sp.symbols('e'): np.e}
    return f_vector.subs(sustituciones).evalf()

# Función para calcular la matriz de información del modelo
def matInf(diseNo : DiseNo, modelo : str, variable : sp.Symbol) -> sp.Matrix:
    """
    Calcula la matriz de información de Fisher para un diseño y modelo dados.

    Args:
        diseNo (DiseNo): Diseño experimental.
        modelo (str): Expresión simbólica del modelo.
        variable (sp.Symbol): Variable simbólica del modelo.

    Returns:
        sp.Matrix: Matriz de información de Fisher.
    """
    f_x = sp.sympify(terminos(modelo,variable))
    matriz_Info = sp.zeros(len(f_x),len(f_x))
    for punto in diseNo.puntos:
        f_val = funcion(f_x, variable, punto.x)
        matriz_Info += punto.p * (f_val * f_val.T)
    return matriz_Info

def funcion_simetrica(M : sp.Matrix, k : int):
    """
    Calcula la suma de los determinantes de todas las submatrices principales de orden k de M.

    Args:
        M (sp.Matrix): Matriz cuadrada.
        k (int): Orden de las submatrices.

    Returns:
        float: Suma de determinantes de submatrices principales de orden k.

    Raises:
        ValueError: Si k no cumple 0 < k <= m.
    """
    if k == 0:
        return 1
    
    m = M.shape[0]

    if not (0 < k <= m):
        raise ValueError("El parámetro k debe cumplir 0 < k <= m")
    
    suma = 0
    for indices in itertools.combinations(range(m), k):
        submatriz = M.extract(indices, indices)
        suma += submatriz.det()
    return suma

def gradiente_caracteristico(M : sp.Matrix, k : int):
    """
    Calcula el gradiente característico de la matriz M para el parámetro k.

    Args:
        M (sp.Matrix): Matriz cuadrada.
        k (int): Parámetro de gradiente.

    Returns:
        sp.Matrix: Matriz del gradiente característico.

    Raises:
        ValueError: Si la matriz no es invertible o k no cumple 0 < k <= m.
    """
    if M.det() == 0:
        raise ValueError("La matriz no tiene inversa.")

    m = M.shape[0]

    if not (0 < k <= m):
        raise ValueError("El parámetro k debe cumplir 0 < k <= m")

    detInv = 1/M.det()

    if k >= m/2:
        tope = m
        i = k
        signo = 1
    else:
        tope = k-1
        i = 0
        signo = -1

    gradiente = sp.zeros(m,m)

    while i <= tope:
        gradiente += detInv*funcion_simetrica(M,m-i)*(-M)**(i-k-1)
        i += 1
    return signo * gradiente

def optimoD(diseNo0 : DiseNo, modelo : str, variable : str, iteraciones : int, intervalo : tuple, subintervalos : tuple = None, cercania : float = None, pmin : float = None, nrefinar : int = None, grafico : bool = False) -> sp.Matrix:
    """
    Optimiza el diseño experimental para un modelo dado utilizando el criterio D-óptimo.
    Args:
        diseNo0 (DiseNo): Instancia inicial del diseño experimental.
        modelo (str): Cadena que representa el modelo estadístico.
        variable (str): Nombre de la variable simbólica utilizada en el modelo.
        iteraciones (int): Número de iteraciones para el proceso de optimización.
        intervalo (tuple): Tupla con los extremos inferior y superior del intervalo de búsqueda.
        subintervalos (int, optional): Número de subintervalos para la búsqueda local en cada iteración. Por defecto es 3.
        cercania (float, optional): Parámetro de cercanía para el refinamiento del diseño. Por defecto es (intervalo[1] - intervalo[0])/25.
        pmin (float, optional): Probabilidad mínima permitida para los puntos del diseño. Por defecto es 0.05.
        grafico (bool, optional): Si se debe mostrar el gráfico de la función. Por defecto es False.

        Returns:
        sp.Matrix: Matriz que representa el diseño experimental optimizado.
        Raises:
            ValueError: Si el extremo superior del intervalo es menor que el inferior.
            ValueError: Si la matriz de información del diseño inicial no es invertible.
        Notas:
            - El método utiliza optimización numérica para maximizar la función de información de Fisher.
            - El diseño se refina periódicamente y al final del proceso para asegurar la calidad del resultado.
            - Se recomienda que el modelo y la variable sean compatibles con sympy.
    """
    diseNo = diseNo0.copia()
    symb = sp.symbols(variable)
    f_x = terminos(modelo, symb)
    m = len(f_x)
    i = 0

    # Comprobacion de datos p.ej intervalo[1] > intervalo[0], o que sea mayor que 0
    if intervalo[1] < intervalo[0]:
        raise ValueError("El extremo superior del intervalo debe ser mayor que el inferior.")

    if subintervalos is None:
        subintervalos = 3

    if cercania is None:
        cercania = (intervalo[1] - intervalo[0])/25
    
    if pmin is None:
        # pmin = 1 / (f_x.shape[0]*(f_x.shape[0] + 1) / 2) #Teorema de caratheodory
        pmin = 0.05

    if nrefinar is None:
        nrefinar = 20

    matInfo = matInf(diseNo, modelo, symb)
    
    if matInfo.det() == 0:
        raise ValueError("La matriz de información no tiene inversa")
    
    matInfoInv = matInfo.inv()

    while i<iteraciones:
        i+=1

        formula = (f_x.T * matInfoInv * f_x)[0]
        
        #TODO Añadir constantes
        #Invertimos el signo de la fórmula para que a la hora de calcular el mínimo se calcule el máximo
        def funcion_a_maximizar_neg(valor):
            return -float(formula.subs({symb : valor, sp.symbols('e'): np.e}).evalf())

        # Distancia de cada subintervalo
        paso = (intervalo[1] - intervalo[0]) / subintervalos

        # Valor provisional de min
        min = funcion_a_maximizar_neg(intervalo[0])
        xn = intervalo[0]
        j = 0
        while j < subintervalos:
            # Calculo de subintervalo actual
            subintervalo = (intervalo[0] + paso * j, intervalo[0] + paso * (j + 1))
            j += 1
            res = 1
            res = minimize_scalar(funcion_a_maximizar_neg, bounds=subintervalo, method='bounded')
            if min > res.fun:
                min = res.fun
                xn = res.x

        # Comprobamos extremo superior
        if min > funcion_a_maximizar_neg(intervalo[1]):
            xn = intervalo[1]

        #Obtiene el valor de la función deshaciendo el cambio de signo
        valor = -funcion_a_maximizar_neg(xn)

        beta = (valor - m) / (m * (valor - 1))

        diseNo.aNadir(xn, beta)

        if i%nrefinar==0:
            # Mostrar la función en caso de indicarlo
            if grafico:
                X = np.linspace(intervalo[0], intervalo[1], 300)
                Y = [funcion_a_maximizar_neg(val) for val in X]

                fig, ax = plt.subplots()
                ax.plot(X, Y)
                def cerrar(event):
                    plt.close(fig)
                fig.canvas.mpl_connect('key_press_event', cerrar)
                plt.show()
            diseNo.refinar(m, cercania, pmin)

        # Calculamos la matriz de información inversa del diseño resultante
        f_valor = funcion(f_x, sp.symbols('x'), xn)
        matInfoInv = (1/(1-beta))*(matInfoInv-(beta*matInfoInv*f_valor*f_valor.T*matInfoInv)/(1-beta+beta*(f_valor.T*matInfoInv*f_valor)[0]))

    diseNo.refinar(m, cercania, pmin)
    return diseNo

def optimo(diseNo0 : DiseNo, modelo : str, variable : str, k : int, iteraciones : int, intervalo : tuple, subintervalos : int = None, cercania : float = None, pmin : float = None, nrefinar : int = None, grafico : bool = False) -> sp.Matrix:
    """
    Optimiza el diseño experimental para un modelo dado utilizando el criterio de gradiente característico.

    Args:
        diseNo0 (DiseNo): Instancia inicial del diseño experimental.
        modelo (str): Cadena que representa el modelo estadístico.
        variable (str): Nombre de la variable simbólica utilizada en el modelo.
        k (int): Parámetro del gradiente característico.
        iteraciones (int): Número de iteraciones para el proceso de optimización.
        intervalo (tuple): Tupla con los extremos inferior y superior del intervalo de búsqueda.
        subintervalos (int, optional): Número de subintervalos para la búsqueda local en cada iteración. Por defecto es 3.
        cercania (float, optional): Parámetro de cercanía para el refinamiento del diseño. Por defecto es (intervalo[1] - intervalo[0])/25.
        pmin (float, optional): Probabilidad mínima permitida para los puntos del diseño. Por defecto es 0.05.
        grafico (bool, optional): Si se debe mostrar el gráfico de la función. Por defecto es False.

    Returns:
        DiseNo: Diseño experimental optimizado.

    Raises:
        ValueError: Si el extremo superior del intervalo es menor que el inferior.
        ValueError: Si la matriz de información del diseño inicial no es invertible.
        ValueError: Si el parámetro k no cumple 0 < k <= m.
    """
    diseNo = diseNo0.copia()
    symb = sp.symbols(variable)
    f_x = terminos(modelo, symb)
    i = 0
    m = len(f_x)

    if not (0 < k <= m):
        raise ValueError("El parámetro k debe cumplir 0 < k <= m")

    # Comprobacion de datos p.ej intervalo[1] > intervalo[0]
    if intervalo[1] < intervalo[0]:
        raise ValueError("El extremo superior del intervalo debe ser mayor que el inferior.")

    if subintervalos is None:
        subintervalos = 3

    if cercania is None:
        cercania = (intervalo[1] - intervalo[0])/25

    if pmin is None:
        # pmin = 1 / (f_x.shape[0]*(f_x.shape[0] + 1) / 2) #Teorema de caratheodory
        pmin = 0.05

    if nrefinar is None:
        nrefinar = 20
    
    while i<iteraciones:
        i+=1
        matInfo = matInf(diseNo, modelo, symb)
        if matInfo.det() == 0:
            raise ValueError("La matriz de información no tiene inversa")

        gradiente = gradiente_caracteristico(matInfo, k)

        formula = (f_x.T * gradiente * f_x)[0]
        
        #TODO Añadir constantes
        def funcion_a_minimizar(valor):
            return float(formula.subs({symb : valor, sp.symbols('e'): np.e}).evalf())

        # Distancia de cada subintervalo
        paso = (intervalo[1] - intervalo[0]) / subintervalos

        # Valor provisional de min
        min = funcion_a_minimizar(intervalo[0])
        xn = intervalo[0]
        j = 0
        while j < subintervalos:
            # Calculo de subintervalo actual
            subintervalo = (intervalo[0] + paso * j, intervalo[0] + paso * (j + 1))
            j += 1
            res = minimize_scalar(funcion_a_minimizar, bounds=subintervalo, method='bounded')
            if min > res.fun:
                min = res.fun
                xn = res.x
        
        #Comprobación de extremo superior
        if min > funcion_a_minimizar(intervalo[1]):
            xn = intervalo[1]

        # Añadimos el punto calculado al diseño
        diseNo.aNadir(xn, 1/(2+i))
        
        if i%nrefinar==0:
            if grafico:
                X = np.linspace(intervalo[0], intervalo[1], 300)
                Y = [funcion_a_minimizar(val) for val in X]

                fig, ax = plt.subplots()
                ax.plot(X, Y)
                def cerrar(event):
                    plt.close(fig)
                fig.canvas.mpl_connect('key_press_event', cerrar)
                plt.show()
            diseNo.refinar(m, cercania, pmin)

    diseNo.refinar(m, cercania, pmin)
    return diseNo
