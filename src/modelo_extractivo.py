"""Módulo de Modelos de Resumen Extractivo - LSTM y TF-IDF"""
# Importa numpy para operaciones numéricas y tensorflow para deep learning
import numpy as np, tensorflow as tf
# Importa keras, el API de alto nivel de TensorFlow
from tensorflow import keras
# Importa layers para construir capas de red neuronal y models para crear modelos
from tensorflow.keras import layers, models
# Importa tipos para anotaciones de tipo (List y Tuple)
from typing import List, Tuple
# Importa el archivo de configuración con hiperparámetros
import config

class ModeloResumenExtractivoLSTM:
    # Clase que implementa un modelo de resumen extractivo usando redes LSTM bidireccionales
    def __init__(self, tamanio_vocabulario: int, dimension_embedding: int = config.DIMENSION_EMBEDDING,
                 dimension_lstm: int = config.DIMENSION_LSTM_EXTRACTIVO, capas_lstm: int = config.CAPAS_LSTM_EXTRACTIVO,
                 tasa_dropout: float = config.TASA_DROPOUT_EXTRACTIVO, longitud_maxima_oracion: int = config.LONGITUD_MAXIMA_ORACION):
        # Constructor que inicializa los hiperparámetros del modelo
        # Guarda el tamaño del vocabulario, dimensión del embedding y dimensión de LSTM
        self.tamanio_vocabulario, self.dimension_embedding, self.dimension_lstm = tamanio_vocabulario, dimension_embedding, dimension_lstm
        # Guarda número de capas LSTM, tasa de dropout, longitud máxima de oración y inicializa modelo en None
        self.capas_lstm, self.tasa_dropout, self.longitud_maxima_oracion, self.modelo = capas_lstm, tasa_dropout, longitud_maxima_oracion, None
        
    def construir_modelo(self) -> keras.Model:
        # Método que construye la arquitectura del modelo de red neuronal
        # Crea la capa de entrada que acepta secuencias de longitud máxima definida
        entrada = layers.Input(shape=(self.longitud_maxima_oracion,), name='entrada_oracion')
        # Capa de embedding que convierte índices de palabras en vectores densos, mask_zero ignora padding
        x = layers.Embedding(self.tamanio_vocabulario, self.dimension_embedding, mask_zero=True, name='embedding')(entrada)
        # Itera para crear múltiples capas LSTM bidireccionales
        for i in range(self.capas_lstm):
            # Capa LSTM bidireccional que procesa secuencias en ambas direcciones (adelante y atrás)
            # return_sequences=True para todas las capas excepto la última
            x = layers.Bidirectional(layers.LSTM(self.dimension_lstm, return_sequences=(i < self.capas_lstm - 1),
                                                 dropout=self.tasa_dropout, recurrent_dropout=self.tasa_dropout,
                                                 name=f'lstm_{i+1}'), name=f'bidirectional_lstm_{i+1}')(x)
            # Aplica dropout entre capas LSTM (excepto después de la última)
            if i < self.capas_lstm - 1:
                x = layers.Dropout(self.tasa_dropout, name=f'dropout_{i+1}')(x)
        # Capa densa con 64 neuronas y activación ReLU para aprender representaciones no lineales
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        # Dropout final para regularización antes de la capa de salida
        x = layers.Dropout(self.tasa_dropout, name='dropout_final')(x)
        # Capa de salida con 1 neurona y activación sigmoid para clasificación binaria (0-1)
        salida = layers.Dense(1, activation='sigmoid', name='salida')(x)
        # Crea el modelo completo conectando entrada y salida
        self.modelo = models.Model(inputs=entrada, outputs=salida, name='modelo_extractivo_lstm')
        # Retorna el modelo construido
        return self.modelo
    
    def compilar(self, tasa_aprendizaje: float = config.TASA_APRENDIZAJE, optimizador: str = config.OPTIMIZADOR):
        # Método que compila el modelo con optimizador, función de pérdida y métricas
        # Verifica que el modelo haya sido construido antes de compilar
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero")
        # Selecciona el optimizador según el parámetro: Adam, RMSprop o SGD
        opt = keras.optimizers.Adam(learning_rate=tasa_aprendizaje) if optimizador.lower() == 'adam' else \
              keras.optimizers.RMSprop(learning_rate=tasa_aprendizaje) if optimizador.lower() == 'rmsprop' else keras.optimizers.SGD(learning_rate=tasa_aprendizaje)
        # Compila el modelo con el optimizador, binary_crossentropy como pérdida y métricas de evaluación
        self.modelo.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
        # Imprime mensaje de confirmación
        print("Modelo extractivo compilado exitosamente")
    
    def resumen(self):
        # Método que muestra un resumen de la arquitectura del modelo
        # Verifica que el modelo exista antes de mostrar el resumen
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero")
        # Retorna el resumen del modelo con información de capas, parámetros, etc.
        return self.modelo.summary()
    
    def entrenar(self, x_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray, x_validacion: np.ndarray, y_validacion: np.ndarray,
                 epocas: int = config.EPOCAS_MAXIMAS, tamanio_lote: int = config.TAMANIO_LOTE, callbacks: List = None) -> keras.callbacks.History:
        # Método que entrena el modelo con datos de entrenamiento y validación
        # Verifica que el modelo esté construido y compilado antes de entrenar
        if self.modelo is None:
            raise ValueError("Debe construir y compilar el modelo primero")
        # Entrena el modelo usando fit con datos de entrenamiento, validación, épocas y tamaño de lote
        # callbacks permite agregar funciones personalizadas durante el entrenamiento
        return self.modelo.fit(x_entrenamiento, y_entrenamiento, validation_data=(x_validacion, y_validacion),
                              epochs=epocas, batch_size=tamanio_lote, callbacks=callbacks, verbose=1)
    
    def predecir(self, oraciones: np.ndarray) -> np.ndarray:
        # Método que realiza predicciones sobre nuevas oraciones
        # Verifica que el modelo exista antes de predecir
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero")
        # Retorna las predicciones del modelo (probabilidades entre 0 y 1)
        return self.modelo.predict(oraciones)
    
    def extraer_oraciones_importantes(self, oraciones: List[str], oraciones_tokenizadas: np.ndarray,
                                     numero_oraciones: int = config.NUMERO_ORACIONES_EXTRAER, umbral: float = 0.5) -> Tuple[List[str], np.ndarray]:
        # Método que extrae las oraciones más importantes de un texto
        # Predice puntuaciones para todas las oraciones y las aplana en un array 1D
        puntuaciones = self.predecir(oraciones_tokenizadas).flatten()
        # Ordena los índices de mayor a menor puntuación usando argsort inverso
        indices_ordenados = np.argsort(puntuaciones)[::-1]
        # Inicializa listas para almacenar oraciones y puntuaciones seleccionadas
        oraciones_seleccionadas, puntuaciones_seleccionadas = [], []
        # Itera sobre los índices ordenados por puntuación
        for idx in indices_ordenados:
            # Detiene si ya se alcanzó el número deseado de oraciones
            if len(oraciones_seleccionadas) >= numero_oraciones:
                break
            # Solo incluye oraciones cuya puntuación supere el umbral
            if puntuaciones[idx] >= umbral:
                oraciones_seleccionadas.append(oraciones[idx])
                puntuaciones_seleccionadas.append(puntuaciones[idx])
        # Retorna las oraciones seleccionadas y sus puntuaciones como tupla
        return oraciones_seleccionadas, np.array(puntuaciones_seleccionadas)
    
    def guardar(self, ruta: str):
        # Método que guarda el modelo entrenado en disco
        # Verifica que exista un modelo antes de guardarlo
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        # Guarda el modelo en la ruta especificada
        self.modelo.save(ruta)
        # Imprime mensaje de confirmación con la ruta
        print(f"Modelo extractivo guardado en {ruta}")
    
    def cargar(self, ruta: str):
        # Método que carga un modelo previamente guardado desde disco
        # Carga el modelo desde la ruta especificada
        self.modelo = keras.models.load_model(ruta)
        # Imprime mensaje de confirmación con la ruta
        print(f"Modelo extractivo cargado desde {ruta}")

class CalculadorPuntuacionTFIDF:
    # Clase que implementa el algoritmo TF-IDF para calcular importancia de oraciones
    def __init__(self):
        # Constructor que inicializa el diccionario IDF vacío y marca como no ajustado
        self.idf, self.ajustado = {}, False
    
    def ajustar(self, documentos: List[List[str]]):
        # Método que calcula los valores IDF (Inverse Document Frequency) del corpus
        # Cuenta el número total de documentos e inicializa diccionario de frecuencias
        num_documentos, frecuencia_documento = len(documentos), {}
        # Itera sobre cada documento del corpus
        for documento in documentos:
            # Itera sobre palabras únicas del documento (usando set para evitar duplicados)
            for palabra in set(documento):
                # Incrementa el contador de documentos que contienen esta palabra
                frecuencia_documento[palabra] = frecuencia_documento.get(palabra, 0) + 1
        # Calcula el valor IDF para cada palabra usando la fórmula log(N / (1 + df))
        for palabra, freq in frecuencia_documento.items():
            self.idf[palabra] = np.log(num_documentos / (1 + freq))
        # Marca el calculador como ajustado
        self.ajustado = True
        # Imprime el número de palabras únicas procesadas
        print(f"TF-IDF ajustado con {len(self.idf)} palabras únicas")
    
    def calcular_puntuacion_oracion(self, oracion: List[str]) -> float:
        # Método que calcula la puntuación TF-IDF de una oración
        # Verifica que el calculador haya sido ajustado con un corpus
        if not self.ajustado:
            raise ValueError("Debe ajustar el calculador primero")
        # Retorna 0 si la oración está vacía
        if not oracion:
            return 0.0
        # Inicializa diccionario para contar frecuencia de términos
        frecuencia_terminos = {}
        # Cuenta cuántas veces aparece cada palabra en la oración
        for palabra in oracion:
            frecuencia_terminos[palabra] = frecuencia_terminos.get(palabra, 0) + 1
        # Normaliza las frecuencias dividiéndolas por la longitud de la oración (calcula TF)
        for palabra in frecuencia_terminos:
            frecuencia_terminos[palabra] /= len(oracion)
        # Calcula y retorna la suma de TF * IDF para todas las palabras
        return sum(tf * self.idf.get(palabra, 0) for palabra, tf in frecuencia_terminos.items())
    
    def extraer_oraciones_importantes(self, oraciones: List[List[str]], numero_oraciones: int = config.NUMERO_ORACIONES_EXTRAER) -> Tuple[List[int], np.ndarray]:
        # Método que extrae las oraciones más importantes usando TF-IDF
        # Calcula la puntuación TF-IDF para cada oración y las almacena en un array
        puntuaciones = np.array([self.calcular_puntuacion_oracion(oracion) for oracion in oraciones])
        # Ordena los índices de mayor a menor puntuación
        indices_ordenados = np.argsort(puntuaciones)[::-1]
        # Retorna los índices y puntuaciones de las N oraciones más importantes
        return indices_ordenados[:numero_oraciones], puntuaciones[indices_ordenados[:numero_oraciones]]

if __name__ == '__main__':
    # Bloque que se ejecuta solo cuando el archivo se ejecuta directamente (no al importarlo)
    print("=== Ejemplo de Modelo Extractivo ===\n")
    # Crea una instancia del modelo LSTM con vocabulario de 10000 palabras
    modelo = ModeloResumenExtractivoLSTM(10000, 128, 64, 2)
    # Construye la arquitectura del modelo
    modelo.construir_modelo()
    # Compila el modelo con optimizador y métricas
    modelo.compilar()
    # Muestra el resumen de la arquitectura del modelo
    modelo.resumen()
    print("\n=== Ejemplo de TF-IDF ===\n")
    # Define documentos de ejemplo como listas de palabras
    docs = [["el", "gato", "está", "en", "el", "tejado"], ["el", "perro", "corre", "por", "el", "parque"], ["el", "gato", "y", "el", "perro", "son", "amigos"]]
    # Crea una instancia del calculador TF-IDF
    calc = CalculadorPuntuacionTFIDF()
    # Ajusta el calculador con los documentos de ejemplo
    calc.ajustar(docs)
    # Itera sobre cada documento y muestra su puntuación TF-IDF
    for i, doc in enumerate(docs):
        print(f"Documento {i+1}: {' '.join(doc)}\nPuntuación TF-IDF: {calc.calcular_puntuacion_oracion(doc):.4f}\n")
