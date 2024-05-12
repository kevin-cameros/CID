import numpy as np

def calcularMedia(datos):
    suma = np.sum(datos)
    return suma / len(datos)

def regresionLineal(x, y):
    n = len(x)

    matrizLineal = np.ones((n, 2))
    matrizLineal[:, 1] = x

    transpuestaX = np.transpose(matrizLineal)

    producto1 = np.dot(transpuestaX, matrizLineal)

    determinante = np.linalg.det(producto1)

    if determinante == 0:
        print("La matriz no es invertible. La regresión lineal no es posible.")
        return

    inversa = np.linalg.inv(producto1)

    producto2 = np.dot(transpuestaX, np.array([y]).T)

    coeficientes = np.dot(inversa, producto2)

    print("Regresión Lineal: y =", coeficientes[0][0], "+", coeficientes[1][0], "* Batch size")

    mediaX = calcularMedia(x)
    mediaY = calcularMedia(y)

    sumaProductos = 0.0
    sumaXCuadrado = 0.0
    sumaYCuadrado = 0.0

    for i in range(n):
        predicho = coeficientes[0][0] + coeficientes[1][0] * x[i]
        sumaProductos += (x[i] - mediaX) * (predicho - mediaY)
        sumaXCuadrado += (x[i] - mediaX) ** 2
        sumaYCuadrado += (y[i] - mediaY) ** 2

    coefCorrelacion = sumaProductos / np.sqrt(sumaXCuadrado * sumaYCuadrado)
    coefDeterminacion = coefCorrelacion ** 2

    print("Coeficiente de Correlación:", coefCorrelacion)
    print("Coeficiente de Determinación:", coefDeterminacion)

def regresionCuadratica(x, y):
    n = len(x)

    matrizCuadratica = np.ones((n, 3))
    matrizCuadratica[:, 1] = x
    matrizCuadratica[:, 2] = np.power(x, 2)

    transpuestaX = np.transpose(matrizCuadratica)

    producto1 = np.dot(transpuestaX, matrizCuadratica)

    determinante = np.linalg.det(producto1)

    if determinante == 0:
        print("La matriz no es invertible. La regresión cuadrática no es posible.")
        return

    inversa = np.linalg.inv(producto1)

    producto2 = np.dot(transpuestaX, np.array([y]).T)

    coeficientes = np.dot(inversa, producto2)

    print("\nRegresión Cuadrática: y =", coeficientes[0][0], "+", coeficientes[1][0], "* Batch size +", coeficientes[2][0], "* Batch size^2")

    mediaX = calcularMedia(x)
    mediaY = calcularMedia(y)

    sumaProductos = 0.0
    sumaXCuadrado = 0.0
    sumaYCuadrado = 0.0

    for i in range(n):
        predicho = coeficientes[0][0] + coeficientes[1][0] * x[i] + coeficientes[2][0] * x[i] ** 2
        sumaProductos += (x[i] - mediaX) * (predicho - mediaY)
        sumaXCuadrado += (x[i] - mediaX) ** 2
        sumaYCuadrado += (y[i] - mediaY) ** 2

    coefCorrelacion = sumaProductos / np.sqrt(sumaXCuadrado * sumaYCuadrado)
    coefDeterminacion = coefCorrelacion ** 2

    print("Coeficiente de Correlación:", coefCorrelacion)
    print("Coeficiente de Determinación:", coefDeterminacion)

def regresionCubica(x, y):
    n = len(x)

    matrizCubica = np.ones((n, 4))
    matrizCubica[:, 1] = x
    matrizCubica[:, 2] = np.power(x, 2)
    matrizCubica[:, 3] = np.power(x, 3)

    transpuestaX = np.transpose(matrizCubica)

    producto1 = np.dot(transpuestaX, matrizCubica)

    inversa = np.linalg.inv(producto1)

    producto2 = np.dot(inversa, transpuestaX)
    coeficientes = np.dot(producto2, np.array([y]).T)

    print("\nRegresión Cúbica: y =", coeficientes[0][0], "+", coeficientes[1][0], "* Batch size +", coeficientes[2][0], "* Batch size^2 +", coeficientes[3][0], "* Batch size^3")

    mediaX = calcularMedia(x)
    mediaY = calcularMedia(y)

    sumaProductos = 0.0
    sumaXCuadrado = 0.0
    sumaYCuadrado = 0.0

    for i in range(n):
        predicho = coeficientes[0][0] + coeficientes[1][0] * x[i] + coeficientes[2][0] * x[i] ** 2 + coeficientes[3][0] * x[i] ** 3
        sumaProductos += (x[i] - mediaX) * (predicho - mediaY)
        sumaXCuadrado += (x[i] - mediaX) ** 2
        sumaYCuadrado += (y[i] - mediaY) ** 2

    coefCorrelacion = sumaProductos / np.sqrt(sumaXCuadrado * sumaYCuadrado)
    coefDeterminacion = coefCorrelacion ** 2

    print("Coeficiente de Correlación:", coefCorrelacion)
    print("Coeficiente de Determinación:", coefDeterminacion)

# Datos de ejemplo
batchSize = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
machineEfficiency = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77 ,96 ,87 ,89 ,60 ,63 ,95 ,61 ,55, 56, 94, 93]

regresionLineal(batchSize, machineEfficiency)
regresionCuadratica(batchSize, machineEfficiency)
regresionCubica(batchSize, machineEfficiency)

