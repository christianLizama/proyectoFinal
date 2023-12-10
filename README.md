# Predicción de Especies de Flores Iris

Este script de Python utiliza un modelo previamente entrenado para predecir la especie de una flor Iris basándose en sus características.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/christianLizama/proyectoFinal.git

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt

## Uso
1. Para predecir la especie de una flor Iris, ejecuta el script test.py proporcionando los valores de las características de la flor como argumentos de línea de comandos con etiquetas correspondientes:
   ```bash 
    python test.py --longitud_sepal 5.1 --ancho_sepal 3.5 --longitud_petalo 1.4 --ancho_petalo 0.2

## Ejecución del Script de Entrenamiento

Al ejecutar el script de entrenamiento, asegúrate de proporcionar los siguientes parámetros:

- `--test_size`: Tamaño del conjunto de prueba.
- `--random_state`: Semilla para reproducibilidad.
- `--n_neighbors`: Número de vecinos para el modelo KNN.

   Por ejemplo:

   ```bash
   python train.py --test_size 0.3 --random_state 42 --n_neighbors 5

### Ejemplo de Resultado

La salida del script mostrará la especie de flor Iris que ha sido predicha basándose en las características proporcionadas:

> La especie predicha para los valores de entrada es: Setosa



