import math

class RegresionLineal:
    def __init__(self, x_values, y_values):
        self.x = x_values
        self.y = y_values
        self.pendiente = None
        self.ordenada_en_el_origen = None
        self.correlacion = None
        self.calcular_regresion_lineal()
        self.calcular_coeficiente_correlacion()

    def calcular_regresion_lineal(self):
        sum_x = sum(self.x)
        sum_y = sum(self.y)
        sum_xy = sum([x * y for x, y in zip(self.x, self.y)])
        sum_x2 = sum([x ** 2 for x in self.x])
        
        self.pendiente = (len(self.x) * sum_xy - sum_x * sum_y) / (len(self.x) * sum_x2 - sum_x ** 2)
        self.ordenada_en_el_origen = (sum_y - self.pendiente * sum_x) / len(self.x)

    def calcular_coeficiente_correlacion(self):
        sum_x = sum(self.x)
        sum_y = sum(self.y)
        sum_xy = sum([x * y for x, y in zip(self.x, self.y)])
        sum_x2 = sum([x ** 2 for x in self.x])
        sum_y2 = sum([y ** 2 for y in self.y])
        
        self.correlacion = (len(self.x) * sum_xy - sum_x * sum_y) / (math.sqrt((len(self.x) * sum_x2 - sum_x ** 2) * (len(self.x) * sum_y2 - sum_y ** 2)))

    def calcular_coeficiente_determinacion(self):
        return self.correlacion ** 2

    def imprimir_resultados(self):
        print("Beta0:", self.ordenada_en_el_origen, "+ Beta1:", self.pendiente)
        print("\nCoeficiente de correlación:", self.correlacion)
        print("Coeficiente de determinación (R^2):", self.calcular_coeficiente_determinacion())

    def predecir_datos_nuevos(self, nuevos_x):
        print("\nPredicciones para nuevos datos:")
        for nuevo_x in nuevos_x:
            prediccion = self.pendiente * nuevo_x + self.ordenada_en_el_origen
            print("Para x =", nuevo_x, ", la predicción es y =", prediccion)

if __name__ == "__main__":
    x = [12, 46, 11, 32, 44, 6, 9, 18, 19]
    y = [65, 762, 866, 1063, 110, 16, 1421, 1406, 1586]

    regresion = RegresionLineal(x, y)

    regresion.imprimir_resultados()

    nuevos_x = [12, 5, 40, 25, 16]
    regresion.predecir_datos_nuevos(nuevos_x)
