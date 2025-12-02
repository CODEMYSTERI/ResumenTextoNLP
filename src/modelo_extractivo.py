"""Módulo de Modelos de Resumen Extractivo - LSTM y TF-IDF"""
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import List, Tuple
import config

class ModeloResumenExtractivoLSTM:
    def __init__(self, tamanio_vocabulario: int, dimension_embedding: int = config.DIMENSION_EMBEDDING,
                 dimension_lstm: int = config.DIMENSION_LSTM_EXTRACTIVO, capas_lstm: int = config.CAPAS_LSTM_EXTRACTIVO,
                 tasa_dropout: float = config.TASA_DROPOUT_EXTRACTIVO, longitud_maxima_oracion: int = config.LONGITUD_MAXIMA_ORACION):
        self.tamanio_vocabulario, self.dimension_embedding, self.dimension_lstm = tamanio_vocabulario, dimension_embedding, dimension_lstm
        self.capas_lstm, self.tasa_dropout, self.longitud_maxima_oracion, self.modelo = capas_lstm, tasa_dropout, longitud_maxima_oracion, None
        
    def construir_modelo(self) -> keras.Model:
        entrada = layers.Input(shape=(self.longitud_maxima_oracion,), name='entrada_oracion')
        x = layers.Embedding(self.tamanio_vocabulario, self.dimension_embedding, mask_zero=True, name='embedding')(entrada)
        for i in range(self.capas_lstm):
            x = layers.Bidirectional(layers.LSTM(self.dimension_lstm, return_sequences=(i < self.capas_lstm - 1),
                                                 dropout=self.tasa_dropout, recurrent_dropout=self.tasa_dropout,
                                                 name=f'lstm_{i+1}'), name=f'bidirectional_lstm_{i+1}')(x)
            if i < self.capas_lstm - 1:
                x = layers.Dropout(self.tasa_dropout, name=f'dropout_{i+1}')(x)
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.tasa_dropout, name='dropout_final')(x)
        salida = layers.Dense(1, activation='sigmoid', name='salida')(x)
        self.modelo = models.Model(inputs=entrada, outputs=salida, name='modelo_extractivo_lstm')
        return self.modelo
    
    def compilar(self, tasa_aprendizaje: float = config.TASA_APRENDIZAJE, optimizador: str = config.OPTIMIZADOR):
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero")
        opt = keras.optimizers.Adam(learning_rate=tasa_aprendizaje) if optimizador.lower() == 'adam' else \
              keras.optimizers.RMSprop(learning_rate=tasa_aprendizaje) if optimizador.lower() == 'rmsprop' else keras.optimizers.SGD(learning_rate=tasa_aprendizaje)
        self.modelo.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
        print("Modelo extractivo compilado exitosamente")
    
    def resumen(self):
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero")
        return self.modelo.summary()
    
    def entrenar(self, x_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray, x_validacion: np.ndarray, y_validacion: np.ndarray,
                 epocas: int = config.EPOCAS_MAXIMAS, tamanio_lote: int = config.TAMANIO_LOTE, callbacks: List = None) -> keras.callbacks.History:
        if self.modelo is None:
            raise ValueError("Debe construir y compilar el modelo primero")
        return self.modelo.fit(x_entrenamiento, y_entrenamiento, validation_data=(x_validacion, y_validacion),
                              epochs=epocas, batch_size=tamanio_lote, callbacks=callbacks, verbose=1)
    
    def predecir(self, oraciones: np.ndarray) -> np.ndarray:
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero")
        return self.modelo.predict(oraciones)
    
    def extraer_oraciones_importantes(self, oraciones: List[str], oraciones_tokenizadas: np.ndarray,
                                     numero_oraciones: int = config.NUMERO_ORACIONES_EXTRAER, umbral: float = 0.5) -> Tuple[List[str], np.ndarray]:
        puntuaciones = self.predecir(oraciones_tokenizadas).flatten()
        indices_ordenados = np.argsort(puntuaciones)[::-1]
        oraciones_seleccionadas, puntuaciones_seleccionadas = [], []
        for idx in indices_ordenados:
            if len(oraciones_seleccionadas) >= numero_oraciones:
                break
            if puntuaciones[idx] >= umbral:
                oraciones_seleccionadas.append(oraciones[idx])
                puntuaciones_seleccionadas.append(puntuaciones[idx])
        return oraciones_seleccionadas, np.array(puntuaciones_seleccionadas)
    
    def guardar(self, ruta: str):
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        self.modelo.save(ruta)
        print(f"Modelo extractivo guardado en {ruta}")
    
    def cargar(self, ruta: str):
        self.modelo = keras.models.load_model(ruta)
        print(f"Modelo extractivo cargado desde {ruta}")

class CalculadorPuntuacionTFIDF:
    def __init__(self):
        self.idf, self.ajustado = {}, False
    
    def ajustar(self, documentos: List[List[str]]):
        num_documentos, frecuencia_documento = len(documentos), {}
        for documento in documentos:
            for palabra in set(documento):
                frecuencia_documento[palabra] = frecuencia_documento.get(palabra, 0) + 1
        for palabra, freq in frecuencia_documento.items():
            self.idf[palabra] = np.log(num_documentos / (1 + freq))
        self.ajustado = True
        print(f"TF-IDF ajustado con {len(self.idf)} palabras únicas")
    
    def calcular_puntuacion_oracion(self, oracion: List[str]) -> float:
        if not self.ajustado:
            raise ValueError("Debe ajustar el calculador primero")
        if not oracion:
            return 0.0
        frecuencia_terminos = {}
        for palabra in oracion:
            frecuencia_terminos[palabra] = frecuencia_terminos.get(palabra, 0) + 1
        for palabra in frecuencia_terminos:
            frecuencia_terminos[palabra] /= len(oracion)
        return sum(tf * self.idf.get(palabra, 0) for palabra, tf in frecuencia_terminos.items())
    
    def extraer_oraciones_importantes(self, oraciones: List[List[str]], numero_oraciones: int = config.NUMERO_ORACIONES_EXTRAER) -> Tuple[List[int], np.ndarray]:
        puntuaciones = np.array([self.calcular_puntuacion_oracion(oracion) for oracion in oraciones])
        indices_ordenados = np.argsort(puntuaciones)[::-1]
        return indices_ordenados[:numero_oraciones], puntuaciones[indices_ordenados[:numero_oraciones]]

if __name__ == '__main__':
    print("=== Ejemplo de Modelo Extractivo ===\n")
    modelo = ModeloResumenExtractivoLSTM(10000, 128, 64, 2)
    modelo.construir_modelo()
    modelo.compilar()
    modelo.resumen()
    print("\n=== Ejemplo de TF-IDF ===\n")
    docs = [["el", "gato", "está", "en", "el", "tejado"], ["el", "perro", "corre", "por", "el", "parque"], ["el", "gato", "y", "el", "perro", "son", "amigos"]]
    calc = CalculadorPuntuacionTFIDF()
    calc.ajustar(docs)
    for i, doc in enumerate(docs):
        print(f"Documento {i+1}: {' '.join(doc)}\nPuntuación TF-IDF: {calc.calcular_puntuacion_oracion(doc):.4f}\n")
