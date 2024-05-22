# Unicamente para resolver raiz cuadrada directamente aunque se puede hacer manual
import math

class DataPoint:
    def __init__(self, height, weight, t_shirt_size=None):
        self.height = height
        self.weight = weight
        self.t_shirt_size = t_shirt_size
    
    def distance(self, other_point):
        return math.sqrt((self.height - other_point.height) ** 2 + (self.weight - other_point.weight) ** 2)

class KNN:
    def __init__(self, k):
        self.k = k
        self.data_points = []

    def fit(self, data_points):
        self.data_points = data_points

    def predict(self, new_point):
        distances = []
        for data_point in self.data_points:
            dist = new_point.distance(data_point)
            distances.append((dist, data_point))
        
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:self.k]

        sizes = {}
        for _, neighbor in nearest_neighbors:
            if neighbor.t_shirt_size in sizes:
                sizes[neighbor.t_shirt_size] += 1
            else:
                sizes[neighbor.t_shirt_size] = 1
        
        predicted_size = max(sizes, key=sizes.get)
        return predicted_size

class Dataset:
    def __init__(self):
        self.data_points = []
    
    def add_data_point(self, height, weight, t_shirt_size):
        self.data_points.append(DataPoint(height, weight, t_shirt_size))
    
    def load_data(self, heights, weights, t_shirt_sizes):
        for height, weight, size in zip(heights, weights, t_shirt_sizes):
            self.add_data_point(height, weight, size)
    
    def get_data_points(self):
        return self.data_points

# Dataset
heights = [158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170]
weights = [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68]
t_shirt_sizes = ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']

# Crear y cargar dataset
dataset = Dataset()
dataset.load_data(heights, weights, t_shirt_sizes)

# Crear modelo KNN
knn = KNN(k=3)
knn.fit(dataset.get_data_points())

# Predecir la talla de una nueva entrada, los valores pueden ser cambiados para otra prediccion
new_customer = DataPoint(170, 70)
predicted_size = knn.predict(new_customer)

print(f'La talla de camiseta predicha para el nuevo cliente es: {predicted_size}')
