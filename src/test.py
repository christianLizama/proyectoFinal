import pickle
import sys
from knn_model import KNNClassifier

# Cargar el modelo entrenado desde el archivo
filename = 'model/knn_model_custom.sav'
loaded_model_custom = pickle.load(open(filename, 'rb'))

# Obtener los valores de las características de la flor desde los argumentos de línea de comandos con etiquetas
if len(sys.argv) != 9 or any(arg.startswith('--') is False for arg in sys.argv[1::2]):
    print("Por favor, proporcione los valores de las características con las etiquetas correspondientes: --longitud_sepal valor --ancho_sepal valor --longitud_petalo valor --ancho_petalo valor")
    sys.exit(1)

arguments = sys.argv[1:]
input_params = {
    arguments[i]: float(arguments[i+1]) for i in range(0, len(arguments), 2)
}

# Verificar y construir el arreglo de características en el orden adecuado
try:
    input_features = [
        input_params['--longitud_sepal'],
        input_params['--ancho_sepal'],
        input_params['--longitud_petalo'],
        input_params['--ancho_petalo']
    ]
except KeyError as e:
    print(f"La etiqueta {e} no está presente o se proporcionaron valores incorrectos.")
    sys.exit(1)

predicted_class_custom = loaded_model_custom.predict([input_features])

# Mapear el valor numérico de la clase a su nombre correspondiente
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
predicted_species_custom = species_mapping[predicted_class_custom[0]]

print("La especie predicha para los valores de entrada es:", predicted_species_custom)
