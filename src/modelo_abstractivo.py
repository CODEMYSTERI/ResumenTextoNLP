"""Módulo de Modelos de Resumen Abstractivo - Seq2Seq con Attention"""
# Importación de numpy para operaciones numéricas y tensorflow para deep learning
import numpy as np, tensorflow as tf
# Importación de keras, el API de alto nivel de TensorFlow
from tensorflow import keras
# Importación de layers (capas) y models (modelos) de Keras
from tensorflow.keras import layers, models
# Importación de tipos para anotaciones de tipo
from typing import Tuple, Optional
# Importación del archivo de configuración con hiperparámetros
import config

# Clase que implementa el mecanismo de atención de Bahdanau para modelos Seq2Seq
class MecanismoAtencionBahdanau(layers.Layer):
    # Constructor que inicializa el mecanismo de atención
    def __init__(self, unidades: int, **kwargs):
        # Llamada al constructor de la clase padre (Layer)
        super(MecanismoAtencionBahdanau, self).__init__(**kwargs)
        # Almacena el número de unidades (dimensión) del mecanismo de atención
        self.unidades = unidades
        # Capa densa W1 para transformar las salidas del encoder
        self.W1 = layers.Dense(unidades, name='atencion_W1')
        # Capa densa W2 para transformar el estado del decoder
        self.W2 = layers.Dense(unidades, name='atencion_W2')
        # Capa densa V para calcular el score de atención (reduce a 1 dimensión)
        self.V = layers.Dense(1, name='atencion_V')
    
    # Método que calcula el vector de contexto y los pesos de atención
    def call(self, estado_decoder, salidas_encoder):
        # Expande las dimensiones del estado del decoder para broadcasting (añade dimensión temporal)
        estado_decoder_expandido = tf.expand_dims(estado_decoder, 1)
        # Calcula el score de atención: V * tanh(W1*encoder_outputs + W2*decoder_state)
        puntuacion = self.V(tf.nn.tanh(self.W1(salidas_encoder) + self.W2(estado_decoder_expandido)))
        # Aplica softmax para normalizar los scores y obtener pesos de atención (suman 1)
        pesos_atencion = tf.nn.softmax(puntuacion, axis=1)
        # Multiplica los pesos de atención por las salidas del encoder (atención ponderada)
        vector_contexto = pesos_atencion * salidas_encoder
        # Suma a lo largo del eje temporal para obtener el vector de contexto final
        vector_contexto = tf.reduce_sum(vector_contexto, axis=1)
        # Retorna el vector de contexto y los pesos de atención
        return vector_contexto, pesos_atencion
    
    # Método para serializar la configuración de la capa (necesario para guardar/cargar el modelo)
    def get_config(self):
        # Obtiene la configuración de la clase padre
        config_dict = super().get_config()
        # Añade el parámetro 'unidades' a la configuración
        config_dict.update({'unidades': self.unidades})
        # Retorna el diccionario de configuración completo
        return config_dict

# Clase que implementa el Encoder del modelo Seq2Seq
class Encoder(layers.Layer):
    # Constructor que inicializa el encoder con sus parámetros
    def __init__(self, tamanio_vocabulario: int, dimension_embedding: int, unidades_encoder: int,
                 capas: int = 2, tasa_dropout: float = 0.3, **kwargs):
        # Llamada al constructor de la clase padre
        super(Encoder, self).__init__(**kwargs)
        # Almacena el tamaño del vocabulario de entrada
        self.tamanio_vocabulario = tamanio_vocabulario
        # Almacena la dimensión de los embeddings (representación vectorial de palabras)
        self.dimension_embedding = dimension_embedding
        # Almacena el número de unidades LSTM en el encoder
        self.unidades_encoder = unidades_encoder
        # Almacena el número de capas LSTM apiladas
        self.capas = capas
        # Almacena la tasa de dropout para regularización (prevenir overfitting)
        self.tasa_dropout = tasa_dropout
        
        # Capa de embedding que convierte índices de palabras en vectores densos, mask_zero ignora padding
        self.embedding = layers.Embedding(tamanio_vocabulario, dimension_embedding, mask_zero=True, name='encoder_embedding')
        # Lista para almacenar las capas LSTM bidireccionales
        self.lstm_layers = []
        # Crea múltiples capas LSTM bidireccionales apiladas
        for i in range(capas):
            # Añade una capa LSTM bidireccional (procesa secuencia en ambas direcciones)
            self.lstm_layers.append(layers.Bidirectional(
                # LSTM con return_sequences=True (retorna toda la secuencia) y return_state=True (retorna estados finales)
                layers.LSTM(unidades_encoder, return_sequences=True, return_state=True,
                           dropout=tasa_dropout, recurrent_dropout=tasa_dropout, name=f'encoder_lstm_{i+1}'),
                name=f'encoder_bidirectional_{i+1}'))
    
    # Método que procesa la entrada a través del encoder
    def call(self, x, training=False):
        # Convierte los índices de palabras en embeddings
        x = self.embedding(x)
        # Listas para almacenar los estados ocultos forward y backward de cada capa
        estados_forward, estados_backward = [], []
        # Procesa la entrada a través de cada capa LSTM bidireccional
        for lstm_layer in self.lstm_layers:
            # Ejecuta la capa LSTM, retorna salidas y estados (h y c) para ambas direcciones
            salidas = lstm_layer(x, training=training)
            # Actualiza x con las salidas de la capa actual (entrada para la siguiente capa)
            x = salidas[0]
            # Almacena los estados forward (h y c) de la dirección hacia adelante
            estados_forward.append((salidas[1], salidas[2]))
            # Almacena los estados backward (h y c) de la dirección hacia atrás
            estados_backward.append((salidas[3], salidas[4]))
        # Concatena los estados ocultos (h) de la última capa en ambas direcciones
        estado_h = layers.Concatenate()([estados_forward[-1][0], estados_backward[-1][0]])
        # Concatena los estados de celda (c) de la última capa en ambas direcciones
        estado_c = layers.Concatenate()([estados_forward[-1][1], estados_backward[-1][1]])
        # Retorna las salidas de todas las posiciones temporales y los estados finales concatenados
        return x, estado_h, estado_c
    
    # Método para serializar la configuración del encoder
    def get_config(self):
        # Obtiene la configuración de la clase padre
        config_dict = super().get_config()
        # Añade todos los parámetros del encoder a la configuración
        config_dict.update({'tamanio_vocabulario': self.tamanio_vocabulario, 'dimension_embedding': self.dimension_embedding,
                           'unidades_encoder': self.unidades_encoder, 'capas': self.capas, 'tasa_dropout': self.tasa_dropout})
        # Retorna el diccionario de configuración completo
        return config_dict

# Clase que implementa el Decoder del modelo Seq2Seq
class Decoder(layers.Layer):
    # Constructor que inicializa el decoder con sus parámetros
    def __init__(self, tamanio_vocabulario: int, dimension_embedding: int, unidades_decoder: int,
                 unidades_encoder: int, capas: int = 2, tasa_dropout: float = 0.3, usar_atencion: bool = True, **kwargs):
        # Llamada al constructor de la clase padre
        super(Decoder, self).__init__(**kwargs)
        # Almacena el tamaño del vocabulario de salida (resumen)
        self.tamanio_vocabulario = tamanio_vocabulario
        # Almacena la dimensión de los embeddings
        self.dimension_embedding = dimension_embedding
        # Almacena el número de unidades LSTM en el decoder
        self.unidades_decoder = unidades_decoder
        # Almacena el número de unidades del encoder (necesario para dimensiones)
        self.unidades_encoder = unidades_encoder
        # Almacena el número de capas LSTM apiladas
        self.capas = capas
        # Almacena la tasa de dropout para regularización
        self.tasa_dropout = tasa_dropout
        # Flag que indica si se usa mecanismo de atención
        self.usar_atencion = usar_atencion
        
        # Capa de embedding para convertir índices de palabras del resumen en vectores
        self.embedding = layers.Embedding(tamanio_vocabulario, dimension_embedding, mask_zero=True, name='decoder_embedding')
        # Si se usa atención, crea el mecanismo de atención de Bahdanau
        if usar_atencion:
            self.atencion = MecanismoAtencionBahdanau(unidades_decoder)
        # Lista para almacenar las capas LSTM del decoder
        self.lstm_layers = []
        # Crea múltiples capas LSTM apiladas
        for i in range(capas):
            # Añade una capa LSTM que retorna secuencias y estados
            self.lstm_layers.append(layers.LSTM(unidades_decoder, return_sequences=True, return_state=True,
                                                dropout=tasa_dropout, recurrent_dropout=tasa_dropout, name=f'decoder_lstm_{i+1}'))
        # Capa densa final que proyecta a la dimensión del vocabulario (para predecir palabras)
        self.fc = layers.Dense(tamanio_vocabulario, name='decoder_output')
    
    # Método que procesa la entrada a través del decoder
    def call(self, x, salidas_encoder, estado_h, estado_c, training=False):
        # Convierte los índices de palabras del resumen en embeddings
        x = self.embedding(x)
        # Si se usa atención, calcula el vector de contexto
        if self.usar_atencion:
            # Calcula el vector de contexto y los pesos de atención usando el mecanismo de Bahdanau
            vector_contexto, pesos_atencion = self.atencion(estado_h, salidas_encoder)
            # Expande dimensiones del vector de contexto para concatenar con embeddings
            vector_contexto = tf.expand_dims(vector_contexto, 1)
            # Repite el vector de contexto para cada paso temporal de la secuencia
            vector_contexto = tf.repeat(vector_contexto, repeats=tf.shape(x)[1], axis=1)
            # Concatena los embeddings con el vector de contexto (información del encoder)
            x = layers.Concatenate(axis=-1)([x, vector_contexto])
        else:
            # Si no se usa atención, los pesos son None
            pesos_atencion = None
        # Procesa la entrada a través de cada capa LSTM del decoder
        for lstm_layer in self.lstm_layers:
            # Ejecuta la capa LSTM con los estados iniciales del encoder
            x, estado_h, estado_c = lstm_layer(x, initial_state=[estado_h, estado_c], training=training)
        # Proyecta las salidas LSTM al tamaño del vocabulario (logits para cada palabra)
        salidas = self.fc(x)
        # Retorna las salidas, los estados finales y los pesos de atención
        return salidas, estado_h, estado_c, pesos_atencion
    
    # Método para serializar la configuración del decoder
    def get_config(self):
        # Obtiene la configuración de la clase padre
        config_dict = super().get_config()
        # Añade todos los parámetros del decoder a la configuración
        config_dict.update({'tamanio_vocabulario': self.tamanio_vocabulario, 'dimension_embedding': self.dimension_embedding,
                           'unidades_decoder': self.unidades_decoder, 'unidades_encoder': self.unidades_encoder,
                           'capas': self.capas, 'tasa_dropout': self.tasa_dropout, 'usar_atencion': self.usar_atencion})
        # Retorna el diccionario de configuración completo
        return config_dict

# Clase principal que encapsula el modelo completo Seq2Seq para resumen abstractivo
class ModeloResumenAbstractivoSeq2Seq:
    # Constructor que inicializa el modelo con todos sus hiperparámetros
    def __init__(self, tamanio_vocabulario_texto: int, tamanio_vocabulario_resumen: int,
                 dimension_embedding: int = config.DIMENSION_EMBEDDING, unidades_encoder: int = config.DIMENSION_ENCODER,
                 unidades_decoder: int = config.DIMENSION_DECODER, capas_encoder: int = config.CAPAS_ENCODER,
                 capas_decoder: int = config.CAPAS_DECODER, tasa_dropout: float = config.TASA_DROPOUT_ABSTRACTIVO,
                 usar_atencion: bool = config.USAR_ATENCION):
        # Almacena el tamaño del vocabulario del texto de entrada
        self.tamanio_vocabulario_texto = tamanio_vocabulario_texto
        # Almacena el tamaño del vocabulario del resumen de salida
        self.tamanio_vocabulario_resumen = tamanio_vocabulario_resumen
        # Almacena la dimensión de los embeddings
        self.dimension_embedding = dimension_embedding
        # Almacena el número de unidades del encoder
        self.unidades_encoder = unidades_encoder
        # Almacena el número de unidades del decoder
        self.unidades_decoder = unidades_decoder
        # Almacena el número de capas del encoder
        self.capas_encoder = capas_encoder
        # Almacena el número de capas del decoder
        self.capas_decoder = capas_decoder
        # Almacena la tasa de dropout
        self.tasa_dropout = tasa_dropout
        # Almacena si se usa mecanismo de atención
        self.usar_atencion = usar_atencion
        # Inicializa el modelo como None (se construirá después)
        self.modelo = None
        # Inicializa el encoder como None
        self.encoder = None
        # Inicializa el decoder como None
        self.decoder = None
    
    # Método que construye la arquitectura completa del modelo Seq2Seq
    def construir_modelo(self) -> keras.Model:
        # Define la entrada del encoder (secuencia de texto de longitud variable)
        entrada_encoder = layers.Input(shape=(None,), name='entrada_texto')
        # Define la entrada del decoder (secuencia de resumen de longitud variable)
        entrada_decoder = layers.Input(shape=(None,), name='entrada_resumen')
        
        # Crea el encoder con los parámetros especificados
        self.encoder = Encoder(self.tamanio_vocabulario_texto, self.dimension_embedding, self.unidades_encoder,
                               self.capas_encoder, self.tasa_dropout)
        # Procesa la entrada del encoder y obtiene salidas y estados finales
        salidas_encoder, estado_h, estado_c = self.encoder(entrada_encoder)
        
        # Crea el decoder (unidades * 2 porque el encoder es bidireccional)
        self.decoder = Decoder(self.tamanio_vocabulario_resumen, self.dimension_embedding, self.unidades_encoder * 2,
                               self.unidades_encoder * 2, self.capas_decoder, self.tasa_dropout, self.usar_atencion)
        # Procesa la entrada del decoder con las salidas y estados del encoder
        salidas_decoder, _, _, _ = self.decoder(entrada_decoder, salidas_encoder, estado_h, estado_c)
        
        # Crea el modelo completo conectando entradas y salidas
        self.modelo = models.Model(inputs=[entrada_encoder, entrada_decoder], outputs=salidas_decoder, name='modelo_seq2seq_atencion')
        # Retorna el modelo construido
        return self.modelo
    
    # Método que compila el modelo con optimizador, función de pérdida y métricas
    def compilar(self, tasa_aprendizaje: float = config.TASA_APRENDIZAJE, optimizador: str = config.OPTIMIZADOR):
        # Verifica que el modelo haya sido construido antes de compilar
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero con construir_modelo()")
        # Selecciona el optimizador según el parámetro (Adam, RMSprop o SGD)
        opt = keras.optimizers.Adam(learning_rate=tasa_aprendizaje) if optimizador.lower() == 'adam' else \
              keras.optimizers.RMSprop(learning_rate=tasa_aprendizaje) if optimizador.lower() == 'rmsprop' else \
              keras.optimizers.SGD(learning_rate=tasa_aprendizaje)
        # Compila el modelo con el optimizador, función de pérdida (crossentropy para clasificación) y métricas
        self.modelo.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Imprime mensaje de confirmación
        print("Modelo abstractivo Seq2Seq compilado exitosamente")
    
    # Método que muestra un resumen de la arquitectura del modelo
    def resumen(self):
        # Verifica que el modelo exista
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero")
        # Retorna el resumen del modelo (capas, parámetros, etc.)
        return self.modelo.summary()
    
    # Método que entrena el modelo con los datos de entrenamiento y validación
    def entrenar(self, x_texto_entrenamiento: np.ndarray, x_resumen_entrenamiento: np.ndarray, y_resumen_entrenamiento: np.ndarray,
                 x_texto_validacion: np.ndarray, x_resumen_validacion: np.ndarray, y_resumen_validacion: np.ndarray,
                 epocas: int = config.EPOCAS_MAXIMAS, tamanio_lote: int = config.TAMANIO_LOTE, callbacks: list = None) -> keras.callbacks.History:
        # Verifica que el modelo esté construido y compilado
        if self.modelo is None:
            raise ValueError("Debe construir y compilar el modelo primero")
        # Entrena el modelo con los datos de entrenamiento y validación
        historia = self.modelo.fit([x_texto_entrenamiento, x_resumen_entrenamiento], y_resumen_entrenamiento,
                                   validation_data=([x_texto_validacion, x_resumen_validacion], y_resumen_validacion),
                                   epochs=epocas, batch_size=tamanio_lote, callbacks=callbacks, verbose=1)
        # Retorna el historial de entrenamiento (pérdidas, métricas por época)
        return historia
    
    # Método que guarda el modelo entrenado en disco
    def guardar(self, ruta: str):
        # Verifica que exista un modelo para guardar
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        # Guarda el modelo completo (arquitectura, pesos y configuración)
        self.modelo.save(ruta)
        # Imprime mensaje de confirmación con la ruta
        print(f"Modelo abstractivo guardado en {ruta}")
    
    # Método que carga un modelo previamente guardado desde disco
    def cargar(self, ruta: str):
        # Carga el modelo especificando las clases personalizadas (necesario para capas custom)
        self.modelo = keras.models.load_model(ruta, custom_objects={'Encoder': Encoder, 'Decoder': Decoder,
                                                                     'MecanismoAtencionBahdanau': MecanismoAtencionBahdanau})
        # Imprime mensaje de confirmación con la ruta
        print(f"Modelo abstractivo cargado desde {ruta}")

# Bloque que se ejecuta solo si este archivo es el script principal
if __name__ == '__main__':
    # Imprime encabezado del ejemplo
    print("=== Ejemplo de Modelo Abstractivo Seq2Seq ===\n")
    # Crea una instancia del modelo con parámetros de ejemplo
    modelo = ModeloResumenAbstractivoSeq2Seq(tamanio_vocabulario_texto=10000, tamanio_vocabulario_resumen=5000,
                                             dimension_embedding=128, unidades_encoder=256, unidades_decoder=256,
                                             capas_encoder=2, capas_decoder=2, usar_atencion=True)
    # Construye la arquitectura del modelo
    modelo.construir_modelo()
    # Compila el modelo con optimizador y función de pérdida
    modelo.compilar()
    # Muestra el resumen de la arquitectura
    modelo.resumen()
    # Imprime mensaje de éxito
    print("\n✅ Modelo Seq2Seq con Attention construido exitosamente!")
