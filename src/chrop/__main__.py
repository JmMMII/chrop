from .diseno import Diseno, optimo, optimoD

def main():
    # Demo mínima: reproduce tu ejemplo de uso.  :contentReference[oaicite:4]{index=4}
    modelo = "x**3 + x**4*e**(-x**3)"
    dis = Diseno([(0.7, 0.5), (1.0, 0.5)])
    optimo(dis, modelo, 'x', k=1, iteraciones=50, intervalo=(0, 1))
    print("Diseño (criterio general k):")
    print(dis)

    dis2 = Diseno([(0.7, 0.5), (1.0, 0.5)])
    optimoD(dis2, modelo, 'x', iteraciones=25, intervalo=(0, 1))
    print("\nDiseño (criterio D):")
    print(dis2)

if __name__ == "__main__":
    main()
