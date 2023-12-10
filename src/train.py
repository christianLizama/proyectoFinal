from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Cargar el conjunto de datos Iris
iris = load_iris()

# Dividir el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

# Crear un modelo KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo KNN en el conjunto de entrenamiento
knn.fit(X_train, y_train)

# Verificar si el directorio 'model' existe, si no, crearlo
directory = 'model'
if not os.path.exists(directory):
    os.makedirs(directory)

# Guardar el modelo entrenado en un archivo
filename = 'model/knn_model.sav'
pickle.dump(knn, open(filename, 'wb'))

# Imprimir la precisión del modelo en el conjunto de prueba
y_pred = knn.predict(X_test)
print("Precisión del modelo: ", accuracy_score(y_test, y_pred))




