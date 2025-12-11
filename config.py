"""
Configuración del Sistema de Resumen Automático
Contiene todos los parámetros e hiperparámetros del proyecto
"""

import os

# ==================== RUTAS DEL PROYECTO ====================
DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_DATOS = os.path.join(DIRECTORIO_BASE, 'datos')
DIRECTORIO_MODELOS = os.path.join(DIRECTORIO_BASE, 'modelos')
DIRECTORIO_SRC = os.path.join(DIRECTORIO_BASE, 'src')

# Archivos de datos
RUTA_DATOS_ENTRENAMIENTO = os.path.join(DIRECTORIO_DATOS, 'articulos_entrenamiento.csv')
RUTA_DATOS_VALIDACION = os.path.join(DIRECTORIO_DATOS, 'articulos_validacion.csv')
RUTA_DATOS_PRUEBA = os.path.join(DIRECTORIO_DATOS, 'articulos_prueba.csv')

# Archivos de modelos
RUTA_MODELO_EXTRACTIVO = os.path.join(DIRECTORIO_MODELOS, 'modelo_extractivo.h5')
RUTA_MODELO_ABSTRACTIVO = os.path.join(DIRECTORIO_MODELOS, 'modelo_abstractivo.h5')
RUTA_TOKENIZADOR_TEXTO = os.path.join(DIRECTORIO_MODELOS, 'tokenizador_texto.pkl')
RUTA_TOKENIZADOR_RESUMEN = os.path.join(DIRECTORIO_MODELOS, 'tokenizador_resumen.pkl')

# ==================== PARÁMETROS DE PREPROCESAMIENTO ====================
# Longitudes máximas
LONGITUD_MAXIMA_TEXTO = 500  # Número máximo de palabras en el texto original
LONGITUD_MAXIMA_RESUMEN = 100  # Número máximo de palabras en el resumen
LONGITUD_MAXIMA_ORACION = 50  # Número máximo de palabras por oración

# Vocabulario
TAMANIO_VOCABULARIO_TEXTO = 50000  # Tamaño del vocabulario para textos
TAMANIO_VOCABULARIO_RESUMEN = 20000  # Tamaño del vocabulario para resúmenes
PALABRA_DESCONOCIDA = '<UNK>'
PALABRA_INICIO = '<START>'
PALABRA_FIN = '<END>'
PALABRA_RELLENO = '<PAD>'

# ==================== HIPERPARÁMETROS DEL MODELO ====================
# Embeddings
DIMENSION_EMBEDDING = 300  # Dimensión de los vectores de palabras
USAR_EMBEDDINGS_PREENTRENADOS = False  # Si usar GloVe/Word2Vec preentrenados

# Modelo Extractivo
DIMENSION_LSTM_EXTRACTIVO = 128
CAPAS_LSTM_EXTRACTIVO = 2
TASA_DROPOUT_EXTRACTIVO = 0.3
NUMERO_ORACIONES_EXTRAER = 5  # Número de oraciones a extraer del texto

# Modelo Abstractivo (Seq2Seq con Attention)
DIMENSION_ENCODER = 256
DIMENSION_DECODER = 256
CAPAS_ENCODER = 2
CAPAS_DECODER = 2
TASA_DROPOUT_ABSTRACTIVO = 0.3
USAR_ATENCION = True  # Usar mecanismo de atención
TIPO_ATENCION = 'bahdanau'  # 'bahdanau' o 'luong'

# ==================== PARÁMETROS DE ENTRENAMIENTO ====================
# General
TAMANIO_LOTE = 32  # Batch size
EPOCAS_MAXIMAS = 50
TASA_APRENDIZAJE = 0.001
OPTIMIZADOR = 'adam'  # 'adam', 'rmsprop', 'sgd'

# Early Stopping
PACIENCIA_EARLY_STOPPING = 5
DELTA_MINIMO = 0.001

# Callbacks
GUARDAR_MEJOR_MODELO = True
REDUCIR_LR_EN_PLATEAU = True
FACTOR_REDUCCION_LR = 0.5
PACIENCIA_REDUCCION_LR = 3

# ==================== PARÁMETROS DE EVALUACIÓN ====================
# Métricas
METRICAS_EVALUACION = ['rouge-1', 'rouge-2', 'rouge-l', 'bleu']
CALCULAR_PERPLEXITY = True

# Generación de resúmenes
ESTRATEGIA_GENERACION = 'beam_search'  # 'greedy', 'beam_search', 'top_k', 'nucleus'
ANCHO_BEAM = 5  # Para beam search
TEMPERATURA = 1.0  # Para sampling
TOP_K = 50  # Para top-k sampling
TOP_P = 0.95  # Para nucleus sampling

# ==================== CONFIGURACIÓN DE REPRODUCIBILIDAD ====================
SEMILLA_ALEATORIA = 42

# ==================== CONFIGURACIÓN DE LOGGING ====================
NIVEL_LOG = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
GUARDAR_LOGS = True
RUTA_LOGS = os.path.join(DIRECTORIO_BASE, 'logs')

# ==================== CONFIGURACIÓN DE VISUALIZACIÓN ====================
ESTILO_GRAFICAS = 'seaborn'
TAMANIO_FIGURA = (12, 6)
DPI_GRAFICAS = 100
GUARDAR_GRAFICAS = True
RUTA_GRAFICAS = os.path.join(DIRECTORIO_BASE, 'resultados')

# ==================== FUNCIONES AUXILIARES ====================
def crear_directorios():
    """Crea los directorios necesarios si no existen"""
    directorios = [
        DIRECTORIO_DATOS,
        DIRECTORIO_MODELOS,
        RUTA_LOGS,
        RUTA_GRAFICAS
    ]
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)

def obtener_configuracion():
    """Retorna un diccionario con toda la configuración"""
    return {
        'rutas': {
            'base': DIRECTORIO_BASE,
            'datos': DIRECTORIO_DATOS,
            'modelos': DIRECTORIO_MODELOS,
        },
        'preprocesamiento': {
            'longitud_maxima_texto': LONGITUD_MAXIMA_TEXTO,
            'longitud_maxima_resumen': LONGITUD_MAXIMA_RESUMEN,
            'tamanio_vocabulario_texto': TAMANIO_VOCABULARIO_TEXTO,
            'tamanio_vocabulario_resumen': TAMANIO_VOCABULARIO_RESUMEN,
        },
        'modelo': {
            'dimension_embedding': DIMENSION_EMBEDDING,
            'dimension_encoder': DIMENSION_ENCODER,
            'dimension_decoder': DIMENSION_DECODER,
            'usar_atencion': USAR_ATENCION,
        },
        'entrenamiento': {
            'tamanio_lote': TAMANIO_LOTE,
            'epocas_maximas': EPOCAS_MAXIMAS,
            'tasa_aprendizaje': TASA_APRENDIZAJE,
        }
    }

if __name__ == '__main__':
    crear_directorios()
    print("Directorios creados exitosamente")
    print(f"Configuración cargada: {obtener_configuracion()}")
