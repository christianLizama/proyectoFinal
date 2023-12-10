from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
