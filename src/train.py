from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# Cargar el conjunto de datos Iris
iris = load_iris()

# Parámetros predeterminados
test_size = None
random_state = None
n_neighbors = None

# Obtener los parámetros de entrada desde la consola o usar valores predeterminados
args = iter(sys.argv[1:])
next_arg = None

for arg in args:
    if arg.startswith("--"):
        next_arg = arg
    elif next_arg == "--test_size":
        test_size = float(arg)
        next_arg = None
    elif next_arg == "--random_state":
        random_state = int(arg)
        next_arg = None
    elif next_arg == "--n_neighbors":
        n_neighbors = int(arg)
        next_arg = None

if test_size is None or random_state is None or n_neighbors is None:
    print("Error: Todos los parámetros (--test_size, --random_state, --n_neighbors) son requeridos.")
    sys.exit(1)

# Dividir el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size, random_state=random_state)

# Crear un modelo KNN con el número de vecinos especificado
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

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

# Generar el informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Calcular y mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Matriz de Confusión')
plt.show()
