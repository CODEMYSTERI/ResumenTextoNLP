"""Módulo de Modelos de Resumen Abstractivo - Seq2Seq con Attention"""
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import config

class MecanismoAtencionBahdanau(layers.Layer):
    def __init__(self, unidades: int, **kwargs):
        super(MecanismoAtencionBahdanau, self).__init__(**kwargs)
        self.unidades = unidades
        self.W1 = layers.Dense(unidades, name='atencion_W1')
        self.W2 = layers.Dense(unidades, name='atencion_W2')
        self.V = layers.Dense(1, name='atencion_V')
    
    def call(self, estado_decoder, salidas_encoder):
        estado_decoder_expandido = tf.expand_dims(estado_decoder, 1)
        puntuacion = self.V(tf.nn.tanh(self.W1(salidas_encoder) + self.W2(estado_decoder_expandido)))
        pesos_atencion = tf.nn.softmax(puntuacion, axis=1)
        vector_contexto = pesos_atencion * salidas_encoder
        vector_contexto = tf.reduce_sum(vector_contexto, axis=1)
        return vector_contexto, pesos_atencion
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({'unidades': self.unidades})
        return config_dict

class Encoder(layers.Layer):
    def __init__(self, tamanio_vocabulario: int, dimension_embedding: int, unidades_encoder: int,
                 capas: int = 2, tasa_dropout: float = 0.3, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.tamanio_vocabulario = tamanio_vocabulario
        self.dimension_embedding = dimension_embedding
        self.unidades_encoder = unidades_encoder
        self.capas = capas
        self.tasa_dropout = tasa_dropout
        
        self.embedding = layers.Embedding(tamanio_vocabulario, dimension_embedding, mask_zero=True, name='encoder_embedding')
        self.lstm_layers = []
        for i in range(capas):
            self.lstm_layers.append(layers.Bidirectional(
                layers.LSTM(unidades_encoder, return_sequences=True, return_state=True,
                           dropout=tasa_dropout, recurrent_dropout=tasa_dropout, name=f'encoder_lstm_{i+1}'),
                name=f'encoder_bidirectional_{i+1}'))
    
    def call(self, x, training=False):
        x = self.embedding(x)
        estados_forward, estados_backward = [], []
        for lstm_layer in self.lstm_layers:
            salidas = lstm_layer(x, training=training)
            x = salidas[0]
            estados_forward.append((salidas[1], salidas[2]))
            estados_backward.append((salidas[3], salidas[4]))
        estado_h = layers.Concatenate()([estados_forward[-1][0], estados_backward[-1][0]])
        estado_c = layers.Concatenate()([estados_forward[-1][1], estados_backward[-1][1]])
        return x, estado_h, estado_c
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({'tamanio_vocabulario': self.tamanio_vocabulario, 'dimension_embedding': self.dimension_embedding,
                           'unidades_encoder': self.unidades_encoder, 'capas': self.capas, 'tasa_dropout': self.tasa_dropout})
        return config_dict

class Decoder(layers.Layer):
    def __init__(self, tamanio_vocabulario: int, dimension_embedding: int, unidades_decoder: int,
                 unidades_encoder: int, capas: int = 2, tasa_dropout: float = 0.3, usar_atencion: bool = True, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.tamanio_vocabulario = tamanio_vocabulario
        self.dimension_embedding = dimension_embedding
        self.unidades_decoder = unidades_decoder
        self.unidades_encoder = unidades_encoder
        self.capas = capas
        self.tasa_dropout = tasa_dropout
        self.usar_atencion = usar_atencion
        
        self.embedding = layers.Embedding(tamanio_vocabulario, dimension_embedding, mask_zero=True, name='decoder_embedding')
        if usar_atencion:
            self.atencion = MecanismoAtencionBahdanau(unidades_decoder)
        self.lstm_layers = []
        for i in range(capas):
            self.lstm_layers.append(layers.LSTM(unidades_decoder, return_sequences=True, return_state=True,
                                                dropout=tasa_dropout, recurrent_dropout=tasa_dropout, name=f'decoder_lstm_{i+1}'))
        self.fc = layers.Dense(tamanio_vocabulario, name='decoder_output')
    
    def call(self, x, salidas_encoder, estado_h, estado_c, training=False):
        x = self.embedding(x)
        if self.usar_atencion:
            vector_contexto, pesos_atencion = self.atencion(estado_h, salidas_encoder)
            vector_contexto = tf.expand_dims(vector_contexto, 1)
            vector_contexto = tf.repeat(vector_contexto, repeats=tf.shape(x)[1], axis=1)
            x = layers.Concatenate(axis=-1)([x, vector_contexto])
        else:
            pesos_atencion = None
        for lstm_layer in self.lstm_layers:
            x, estado_h, estado_c = lstm_layer(x, initial_state=[estado_h, estado_c], training=training)
        salidas = self.fc(x)
        return salidas, estado_h, estado_c, pesos_atencion
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({'tamanio_vocabulario': self.tamanio_vocabulario, 'dimension_embedding': self.dimension_embedding,
                           'unidades_decoder': self.unidades_decoder, 'unidades_encoder': self.unidades_encoder,
                           'capas': self.capas, 'tasa_dropout': self.tasa_dropout, 'usar_atencion': self.usar_atencion})
        return config_dict

class ModeloResumenAbstractivoSeq2Seq:
    def __init__(self, tamanio_vocabulario_texto: int, tamanio_vocabulario_resumen: int,
                 dimension_embedding: int = config.DIMENSION_EMBEDDING, unidades_encoder: int = config.DIMENSION_ENCODER,
                 unidades_decoder: int = config.DIMENSION_DECODER, capas_encoder: int = config.CAPAS_ENCODER,
                 capas_decoder: int = config.CAPAS_DECODER, tasa_dropout: float = config.TASA_DROPOUT_ABSTRACTIVO,
                 usar_atencion: bool = config.USAR_ATENCION):
        self.tamanio_vocabulario_texto = tamanio_vocabulario_texto
        self.tamanio_vocabulario_resumen = tamanio_vocabulario_resumen
        self.dimension_embedding = dimension_embedding
        self.unidades_encoder = unidades_encoder
        self.unidades_decoder = unidades_decoder
        self.capas_encoder = capas_encoder
        self.capas_decoder = capas_decoder
        self.tasa_dropout = tasa_dropout
        self.usar_atencion = usar_atencion
        self.modelo = None
        self.encoder = None
        self.decoder = None
    
    def construir_modelo(self) -> keras.Model:
        entrada_encoder = layers.Input(shape=(None,), name='entrada_texto')
        entrada_decoder = layers.Input(shape=(None,), name='entrada_resumen')
        
        self.encoder = Encoder(self.tamanio_vocabulario_texto, self.dimension_embedding, self.unidades_encoder,
                               self.capas_encoder, self.tasa_dropout)
        salidas_encoder, estado_h, estado_c = self.encoder(entrada_encoder)
        
        self.decoder = Decoder(self.tamanio_vocabulario_resumen, self.dimension_embedding, self.unidades_encoder * 2,
                               self.unidades_encoder * 2, self.capas_decoder, self.tasa_dropout, self.usar_atencion)
        salidas_decoder, _, _, _ = self.decoder(entrada_decoder, salidas_encoder, estado_h, estado_c)
        
        self.modelo = models.Model(inputs=[entrada_encoder, entrada_decoder], outputs=salidas_decoder, name='modelo_seq2seq_atencion')
        return self.modelo
    
    def compilar(self, tasa_aprendizaje: float = config.TASA_APRENDIZAJE, optimizador: str = config.OPTIMIZADOR):
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero con construir_modelo()")
        opt = keras.optimizers.Adam(learning_rate=tasa_aprendizaje) if optimizador.lower() == 'adam' else \
              keras.optimizers.RMSprop(learning_rate=tasa_aprendizaje) if optimizador.lower() == 'rmsprop' else \
              keras.optimizers.SGD(learning_rate=tasa_aprendizaje)
        self.modelo.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Modelo abstractivo Seq2Seq compilado exitosamente")
    
    def resumen(self):
        if self.modelo is None:
            raise ValueError("Debe construir el modelo primero")
        return self.modelo.summary()
    
    def entrenar(self, x_texto_entrenamiento: np.ndarray, x_resumen_entrenamiento: np.ndarray, y_resumen_entrenamiento: np.ndarray,
                 x_texto_validacion: np.ndarray, x_resumen_validacion: np.ndarray, y_resumen_validacion: np.ndarray,
                 epocas: int = config.EPOCAS_MAXIMAS, tamanio_lote: int = config.TAMANIO_LOTE, callbacks: list = None) -> keras.callbacks.History:
        if self.modelo is None:
            raise ValueError("Debe construir y compilar el modelo primero")
        historia = self.modelo.fit([x_texto_entrenamiento, x_resumen_entrenamiento], y_resumen_entrenamiento,
                                   validation_data=([x_texto_validacion, x_resumen_validacion], y_resumen_validacion),
                                   epochs=epocas, batch_size=tamanio_lote, callbacks=callbacks, verbose=1)
        return historia
    
    def guardar(self, ruta: str):
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        self.modelo.save(ruta)
        print(f"Modelo abstractivo guardado en {ruta}")
    
    def cargar(self, ruta: str):
        self.modelo = keras.models.load_model(ruta, custom_objects={'Encoder': Encoder, 'Decoder': Decoder,
                                                                     'MecanismoAtencionBahdanau': MecanismoAtencionBahdanau})
        print(f"Modelo abstractivo cargado desde {ruta}")

if __name__ == '__main__':
    print("=== Ejemplo de Modelo Abstractivo Seq2Seq ===\n")
    modelo = ModeloResumenAbstractivoSeq2Seq(tamanio_vocabulario_texto=10000, tamanio_vocabulario_resumen=5000,
                                             dimension_embedding=128, unidades_encoder=256, unidades_decoder=256,
                                             capas_encoder=2, capas_decoder=2, usar_atencion=True)
    modelo.construir_modelo()
    modelo.compilar()
    modelo.resumen()
    print("\n✅ Modelo Seq2Seq con Attention construido exitosamente!")
