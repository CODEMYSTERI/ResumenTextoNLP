import os

DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_DATOS = os.path.join(DIRECTORIO_BASE, 'datos')
DIRECTORIO_MODELOS = os.path.join(DIRECTORIO_BASE, 'modelos')
DIRECTORIO_SRC = os.path.join(DIRECTORIO_BASE, 'src')

RUTA_MODELO_EXTRACTIVO = os.path.join(DIRECTORIO_MODELOS, 'modelo_extractivo.h5')
RUTA_MODELO_ABSTRACTIVO = os.path.join(DIRECTORIO_MODELOS, 'modelo_abstractivo.h5')
RUTA_TOKENIZADOR_TEXTO = os.path.join(DIRECTORIO_MODELOS, 'tokenizador_texto.pkl')
RUTA_TENIZADOR_RESUMEN = os.path.join(DIRECTORIO_MODELOS, 'tokenizador_resumen.pkl')

RUTA_DATOS_ENTRENAMIENTO = os.path.join(DIRECTORIO_DATOS, 'datos_entrenamiento.csv')
RUTA_DATOS_VALIDACION = os.path.join(DIRECTORIO_DATOS, 'datos_validacion.csv')
RUTA_DATOS_PRUEBA = os.path.join(DIRECTORIO_DATOS, 'datos_prueba.csv')

LONGITUD_MAXIMA_TEXTO = 500
LONGITUD_MAXIMA_RESUMEN = 100
LONGITUD_MAXIMA_ORACION = 50

TAMANO_VOCABULARIO_TEXTO = 50000
TAMANO_VOCABULARIO_RESUMEN = 20000

PALABRA_DESCONOCIDA = '<UNK>'
PALABRA_INICIO = '<START>'
PALABRA_FIN = '<END>'
PALABRA_RELLENO = '<PAD>'

DIMENSION_EMBEDDING = 300
USAR_EMBEDDING_PREENTRENADORES = False

DIMENSION_LSTM_EXTRACTIVO = 128
CAPAS_LSTM_EXTRACTIVO = 2
TAZA_DROPOUT_EXTRACTIVO = 0.3
NUMERO_ORACIONES_EXTRAER = 5

DIMENSION_ENCODER = 256
DIMENSION_DECODER = 256
CAPAS_ENCODER = 2
CAPAS_DECODER = 2
TAZA_DROPOUT_ABSTRACCION = 0.3
RACTIVO = 0.3

USAR_ATENCION = True
TIPO_ATENCION = 'bahdanau'

TAMANO_LOTE = 32
EPOCAS_MAXIMAS = 50
TAZA_APRENDIZAJE = 0.001
OPTIMIZADOR = 'adam'

POTENCIA_EARLY_STTOPING = 5
DELTA_MINIMO = 0.001

GUARDAR_MEJOR_MODELO = True
REDUCIR_LR_EN_LATEAU = True
FACTOR_REDUCCION_LR = 0.5
PACIENCIA_REDUCCION_LR = 3

# Métricas
METRICAS_EVALUACION = ['rouge-1', 'rouge-2', 'rouge-l', 'bleu']
CALCULAR_PERPLEXITY = True

# Generación de resúmenes
ESTRATEGIA_GENERACION = 'beam_search'  # 'greedy', 'beam_search', 'top_k', 'nucleus'
ANCHO_BEAM = 5  # Para beam search
TEMPERATURA = 1.0  # Para sampling
TOP_K = 50  # Para top-k sampling
TOP_P = 0.95  # Para nucleus sampling

SEMILLA_ALEATORIA = 42
NIVEL_LOG = 'INFO'
GUARDAR_LOGS = True
RUTA_LOGS = os.path.join(DIRECTORIO_BASE, 'logs')
ESTILO_GRAFICAS = 'seaborn'
TAMANO_FIGURA = (12, 6)
DPI_GRAFICAS = 100
GUADAR_GRAFICAS = True
RUTA_GRAFICAS = os.path.join(DIRECTORIO_BASE, 'resultados')

def crear_directorios():
    directorios = [DIRECTORIO_DATOS, DIRECTORIO_MODELOS, RUTA_GRAFICAS, RUTA_LOGS]
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)

def obtener_configuraciones():
    return {
        'rutas':{
            'base':DIRECTORIO_BASE,
            'datos':DIRECTORIO_DATOS,
            'modelo':DIRECTORIO_MODELOS,
        },
        'preprocesamiento':{
            'logitud_maxima_texto':LONGITUD_MAXIMA_TEXTO,
            'longitud_maxima_resumen':LONGITUD_MAXIMA_RESUMEN,
            'tamano_vocabulario_texto':TAMANO_VOCABULARIO_TEXTO,
            'tamano_vocaulario_resumen':TAMANO_VOCABULARIO_RESUMEN,
        },
        'modelo':{
            'dimension_embedding':DIMENSION_EMBEDDING,
            'dimension_encoder':DIMENSION_ENCODER,
            'dimension_decoder':DIMENSION_DECODER,
            'usar_atencion':USAR_ATENCION,
        },
        'entrenamiento':{
            'tamano_lote':TAMANO_LOTE,
            'epocas_maximas':EPOCAS_MAXIMAS,
            'taza_aprendizaje':TAZA_APRENDIZAJE,
        }
    }
if __name__ == '__main__':
    crear_directorios()
    print("Directorios creados con exito...")
    print(f"configuracion cargada: {obtener_configuraciones()}")