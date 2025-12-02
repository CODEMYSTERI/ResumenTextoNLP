"""Script de Entrenamiento del Sistema de Resumen Automático"""
import os, sys, numpy as np, pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.procesamiento import LimpiadorTexto, TokenizadorTexto, PreparadorDatos
from src.modelo_abstractivo import ModeloResumenAbstractivoSeq2Seq

def cargar_datos(ruta_csv: str):
    if not os.path.exists(ruta_csv):
        print(f"⚠️  Archivo no encontrado: {ruta_csv}")
        return None
    df = pd.read_csv(ruta_csv)
    print(f"✅ Datos cargados: {len(df)} ejemplos")
    return df

def crear_callbacks_entrenamiento(nombre_modelo: str, ruta_guardado: str):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.PACIENCIA_EARLY_STOPPING,
                                     min_delta=config.DELTA_MINIMO, restore_best_weights=True, verbose=1)
    ]
    if config.GUARDAR_MEJOR_MODELO:
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath=ruta_guardado, monitor='val_loss',
                                                         save_best_only=True, verbose=1))
    if config.REDUCIR_LR_EN_PLATEAU:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config.FACTOR_REDUCCION_LR,
                                                           patience=config.PACIENCIA_REDUCCION_LR, min_lr=1e-7, verbose=1))
    if config.GUARDAR_LOGS:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(config.RUTA_LOGS, nombre_modelo), histogram_freq=1))
    return callbacks


def entrenar_modelo_abstractivo():
    print("\n" + "="*60 + "\n🚀 ENTRENAMIENTO DEL MODELO ABSTRACTIVO (SEQ2SEQ)\n" + "="*60 + "\n")
    
    print("📂 Cargando datos de entrenamiento...")
    df_train = cargar_datos(config.RUTA_DATOS_ENTRENAMIENTO)
    df_val = cargar_datos(config.RUTA_DATOS_VALIDACION)
    
    if df_train is None:
        print("❌ No se encontraron datos. Generando datos de ejemplo...")
        generar_datos_ejemplo()
        df_train = cargar_datos(config.RUTA_DATOS_ENTRENAMIENTO)
        df_val = cargar_datos(config.RUTA_DATOS_VALIDACION)
    elif df_val is None:
        print("⚠️  Creando validación desde entrenamiento...")
        df_val = df_train.sample(frac=0.2, random_state=config.SEMILLA_ALEATORIA)
        df_train = df_train.drop(df_val.index)
        df_val.to_csv(config.RUTA_DATOS_VALIDACION, index=False)
        df_train.to_csv(config.RUTA_DATOS_ENTRENAMIENTO, index=False)
        print(f"✅ Datos divididos: {len(df_train)} entrenamiento, {len(df_val)} validación")
    
    textos_train, resumenes_train = df_train['texto'].tolist(), df_train['resumen'].tolist()
    textos_val, resumenes_val = df_val['texto'].tolist(), df_val['resumen'].tolist()
    print(f"📊 Entrenamiento: {len(textos_train)} | Validación: {len(textos_val)}\n")
    
    print("🔧 Preparando datos...")
    limpiador = LimpiadorTexto(mantener_acentos=True, minusculas=True)
    tokenizador_texto = TokenizadorTexto(config.TAMANIO_VOCABULARIO_TEXTO, config.PALABRA_DESCONOCIDA,
                                         config.PALABRA_INICIO, config.PALABRA_FIN, config.PALABRA_RELLENO)
    tokenizador_resumen = TokenizadorTexto(config.TAMANIO_VOCABULARIO_RESUMEN, config.PALABRA_DESCONOCIDA,
                                           config.PALABRA_INICIO, config.PALABRA_FIN, config.PALABRA_RELLENO)
    
    print("📝 Construyendo vocabularios...")
    tokenizador_texto.ajustar([limpiador.limpiar_texto(t) for t in textos_train])
    tokenizador_resumen.ajustar([limpiador.limpiar_texto(r) for r in resumenes_train])
    
    preparador = PreparadorDatos(limpiador, tokenizador_texto, tokenizador_resumen)
    X_train, decoder_input_train, decoder_target_train = preparador.preparar_datos(
        textos_train, resumenes_train, config.LONGITUD_MAXIMA_TEXTO, config.LONGITUD_MAXIMA_RESUMEN)
    X_val, decoder_input_val, decoder_target_val = preparador.preparar_datos(
        textos_val, resumenes_val, config.LONGITUD_MAXIMA_TEXTO, config.LONGITUD_MAXIMA_RESUMEN)
    
    print(f"✅ Datos: Textos {X_train.shape} | Decoder input {decoder_input_train.shape}\n")
    
    print("🏗️  Construyendo modelo Seq2Seq con Attention...")
    modelo = ModeloResumenAbstractivoSeq2Seq(
        len(tokenizador_texto.palabra_a_indice), len(tokenizador_resumen.palabra_a_indice),
        config.DIMENSION_EMBEDDING, config.DIMENSION_ENCODER, config.DIMENSION_DECODER,
        config.CAPAS_ENCODER, config.CAPAS_DECODER, config.TASA_DROPOUT_ABSTRACTIVO, config.USAR_ATENCION)
    
    modelo.construir_modelo()
    modelo.compilar(config.TASA_APRENDIZAJE, config.OPTIMIZADOR)
    print("\n📋 Resumen del modelo:")
    modelo.resumen()
    
    callbacks = crear_callbacks_entrenamiento('modelo_abstractivo', config.RUTA_MODELO_ABSTRACTIVO)
    
    print("\n🎯 Iniciando entrenamiento...")
    historia = modelo.entrenar(X_train, decoder_input_train, decoder_target_train,
                               X_val, decoder_input_val, decoder_target_val,
                               config.EPOCAS_MAXIMAS, config.TAMANIO_LOTE, callbacks)
    
    print("\n💾 Guardando modelo y tokenizadores...")
    modelo.guardar(config.RUTA_MODELO_ABSTRACTIVO)
    tokenizador_texto.guardar(config.RUTA_TOKENIZADOR_TEXTO)
    tokenizador_resumen.guardar(config.RUTA_TOKENIZADOR_RESUMEN)
    
    visualizar_entrenamiento(historia, 'Modelo Abstractivo')
    print("\n✅ ¡Entrenamiento completado exitosamente!")
    return modelo, historia


def visualizar_entrenamiento(historia, titulo: str):
    fig, axes = plt.subplots(1, 2, figsize=config.TAMANIO_FIGURA)
    axes[0].plot(historia.history['loss'], label='Entrenamiento')
    axes[0].plot(historia.history['val_loss'], label='Validación')
    axes[0].set_title(f'{titulo} - Pérdida')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(historia.history['accuracy'], label='Entrenamiento')
    axes[1].plot(historia.history['val_accuracy'], label='Validación')
    axes[1].set_title(f'{titulo} - Precisión')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Precisión')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    
    if config.GUARDAR_GRAFICAS:
        ruta_grafica = os.path.join(config.RUTA_GRAFICAS, f'{titulo.lower().replace(" ", "_")}_entrenamiento.png')
        plt.savefig(ruta_grafica, dpi=config.DPI_GRAFICAS, bbox_inches='tight')
        print(f"📊 Gráfica guardada en: {ruta_grafica}")
    plt.show()

def generar_datos_ejemplo():
    print("🔨 Generando datos de ejemplo...")
    datos_ejemplo = {
        'texto': [
            "La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Estos sistemas pueden aprender de la experiencia, ajustarse a nuevas entradas y realizar tareas similares a las humanas. El aprendizaje automático y el aprendizaje profundo son subcampos importantes de la IA.",
            "El cambio climático es uno de los mayores desafíos que enfrenta la humanidad en el siglo XXI. Las temperaturas globales están aumentando debido a las emisiones de gases de efecto invernadero. Esto está causando el derretimiento de los glaciares, el aumento del nivel del mar y eventos climáticos extremos más frecuentes.",
            "Python es un lenguaje de programación de alto nivel, interpretado y de propósito general. Su filosofía de diseño enfatiza la legibilidad del código con el uso de una sintaxis clara. Python soporta múltiples paradigmas de programación, incluyendo programación orientada a objetos, imperativa y funcional.",
            "Las redes neuronales son modelos computacionales inspirados en el cerebro humano. Están compuestas por capas de nodos interconectados que procesan información. Cada conexión tiene un peso que se ajusta durante el entrenamiento. Las redes neuronales profundas han revolucionado campos como la visión por computadora y el procesamiento del lenguaje natural.",
            "El aprendizaje profundo es una técnica de machine learning que utiliza redes neuronales con múltiples capas. Estas redes pueden aprender representaciones jerárquicas de los datos. El deep learning ha logrado avances impresionantes en reconocimiento de imágenes, procesamiento de voz y traducción automática.",
        ],
        'resumen': [
            "La IA crea sistemas que realizan tareas que requieren inteligencia humana mediante aprendizaje automático.",
            "El cambio climático causa aumento de temperaturas, derretimiento de glaciares y eventos climáticos extremos.",
            "Python es un lenguaje de programación de alto nivel con sintaxis clara y múltiples paradigmas.",
            "Las redes neuronales son modelos inspirados en el cerebro que procesan información mediante capas interconectadas.",
            "El deep learning usa redes neuronales profundas para aprender representaciones jerárquicas de datos.",
        ]
    }
    df_completo = pd.DataFrame(datos_ejemplo)
    df_train = df_completo.sample(frac=0.8, random_state=config.SEMILLA_ALEATORIA)
    df_val = df_completo.drop(df_train.index)
    config.crear_directorios()
    df_train.to_csv(config.RUTA_DATOS_ENTRENAMIENTO, index=False)
    df_val.to_csv(config.RUTA_DATOS_VALIDACION, index=False)
    print(f"✅ Datos generados: {len(df_train)} entrenamiento, {len(df_val)} validación")

if __name__ == '__main__':
    np.random.seed(config.SEMILLA_ALEATORIA)
    config.crear_directorios()
    modelo, historia = entrenar_modelo_abstractivo()
    print("\n" + "="*60 + "\n🎉 ¡ENTRENAMIENTO FINALIZADO!\n" + "="*60)
