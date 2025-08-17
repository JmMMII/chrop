from .diseno import DiseNo, optimo, optimoD

def main():
    # Demo mínima: reproduce tu ejemplo de uso.  :contentReference[oaicite:4]{index=4}
    modelo = "x**3 + x**4*e**(-x**3)"
    dis = DiseNo([(0.7, 0.5), (1.0, 0.5)])
    res = optimo(dis, modelo, 'x', k=1, iteraciones=50, intervalo=(0, 1))
    print("Diseño (criterio general k):")
    print(res)

    dis2 = DiseNo([(0.7, 0.5), (1.0, 0.5)])
    res2 = optimoD(dis2, modelo, 'x', iteraciones=25, intervalo=(0, 1))
    print("\nDiseño (criterio D):")
    print(res2)

if __name__ == "__main__":
    main()
