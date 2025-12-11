"""Módulo de Predicción y Generación de Resúmenes"""
# Importación de librerías necesarias para el sistema operativo, rutas y arrays numéricos
import os, sys, numpy as np, tensorflow as tf
from typing import List

# Agregar el directorio raíz al path para permitir importaciones relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Importar configuración global y módulos de preprocesamiento
import config
from src.preprocesamiento import LimpiadorTexto, TokenizadorTexto

class GeneradorResumenes:
    """Clase principal para generar resúmenes automáticos de textos"""
    
    def __init__(self, ruta_modelo: str = config.RUTA_MODELO_ABSTRACTIVO,
                 ruta_tokenizador_texto: str = config.RUTA_TOKENIZADOR_TEXTO,
                 ruta_tokenizador_resumen: str = config.RUTA_TOKENIZADOR_RESUMEN):
        """
        Inicializa el generador de resúmenes cargando modelo y tokenizadores
        
        Args:
            ruta_modelo: Ruta al archivo del modelo entrenado
            ruta_tokenizador_texto: Ruta al tokenizador del texto de entrada
            ruta_tokenizador_resumen: Ruta al tokenizador del resumen de salida
        """
        # Inicializar limpiador de texto con configuración específica
        self.limpiador = LimpiadorTexto(mantener_acentos=True, minusculas=True)
        # Cargar tokenizadores previamente entrenados desde disco
        print("Cargando tokenizadores...")
        self.tokenizador_texto = TokenizadorTexto.cargar(ruta_tokenizador_texto)
        self.tokenizador_resumen = TokenizadorTexto.cargar(ruta_tokenizador_resumen)
        
        # Cargar modelo de red neuronal con objetos personalizados
        print("Cargando modelo...")
        from src.modelo_abstractivo import Encoder, Decoder, MecanismoAtencionBahdanau
        self.modelo = tf.keras.models.load_model(ruta_modelo, custom_objects={
            'Encoder': Encoder, 'Decoder': Decoder, 'MecanismoAtencionBahdanau': MecanismoAtencionBahdanau}, compile=False)
        print("Generador de resumenes listo!\n")
    
    def preprocesar_texto(self, texto: str) -> np.ndarray:
        """
        Preprocesa el texto de entrada para alimentar al modelo
        
        Args:
            texto: Texto original a procesar
            
        Returns:
            Array numpy con la secuencia tokenizada y rellenada
        """
        # Limpiar el texto eliminando caracteres especiales y normalizando
        texto_limpio = self.limpiador.limpiar_texto(texto)
        # Convertir texto a secuencia de índices numéricos
        secuencia = self.tokenizador_texto.texto_a_secuencia(texto_limpio)
        # Rellenar o truncar la secuencia a la longitud máxima configurada
        secuencia_rellenada = self.tokenizador_texto.rellenar_secuencias([secuencia], config.LONGITUD_MAXIMA_TEXTO)
        return secuencia_rellenada
    
    def generar_resumen_greedy(self, texto: str, longitud_maxima: int = None) -> str:
        """
        Genera un resumen usando estrategia greedy (selecciona siempre el token más probable)
        
        Args:
            texto: Texto original a resumir
            longitud_maxima: Longitud máxima del resumen en tokens
            
        Returns:
            Resumen generado como cadena de texto
        """
        # Usar longitud máxima por defecto si no se especifica
        if longitud_maxima is None:
            longitud_maxima = config.LONGITUD_MAXIMA_RESUMEN
        # Preprocesar el texto de entrada
        texto_procesado = self.preprocesar_texto(texto)
        # Obtener índices de tokens especiales de inicio y fin
        idx_inicio = self.tokenizador_resumen.palabra_a_indice[config.PALABRA_INICIO]
        idx_fin = self.tokenizador_resumen.palabra_a_indice[config.PALABRA_FIN]
        # Inicializar secuencia del decoder con token de inicio
        secuencia_decoder = [idx_inicio]
        
        # Generar tokens uno por uno hasta alcanzar longitud máxima o token de fin
        for _ in range(longitud_maxima):
            # Preparar entrada para el decoder
            decoder_input = np.array([secuencia_decoder])
            # Obtener predicciones del modelo
            predicciones = self.modelo.predict([texto_procesado, decoder_input], verbose=0)
            # Seleccionar el token con mayor probabilidad
            siguiente_token = np.argmax(predicciones[0, -1, :])
            # Detener si se genera el token de fin
            if siguiente_token == idx_fin:
                break
            # Agregar token generado a la secuencia
            secuencia_decoder.append(siguiente_token)
        
        # Convertir secuencia de índices a texto legible
        resumen = self.tokenizador_resumen.secuencia_a_texto(secuencia_decoder, eliminar_tokens_especiales=True)
        return resumen
    
    def generar_resumen_beam_search(self, texto: str, ancho_beam: int = config.ANCHO_BEAM, longitud_maxima: int = None) -> str:
        """
        Genera un resumen usando beam search (mantiene múltiples hipótesis candidatas)
        
        Args:
            texto: Texto original a resumir
            ancho_beam: Número de hipótesis a mantener en cada paso
            longitud_maxima: Longitud máxima del resumen en tokens
            
        Returns:
            Resumen generado como cadena de texto
        """
        # Usar longitud máxima por defecto si no se especifica
        if longitud_maxima is None:
            longitud_maxima = config.LONGITUD_MAXIMA_RESUMEN
        # Preprocesar el texto de entrada
        texto_procesado = self.preprocesar_texto(texto)
        # Obtener índices de tokens especiales de inicio y fin
        idx_inicio = self.tokenizador_resumen.palabra_a_indice[config.PALABRA_INICIO]
        idx_fin = self.tokenizador_resumen.palabra_a_indice[config.PALABRA_FIN]
        # Inicializar beam con una hipótesis: secuencia de inicio y puntuación 0
        beam = [([idx_inicio], 0.0)]
        
        # Expandir el beam iterativamente hasta alcanzar longitud máxima
        for _ in range(longitud_maxima):
            nuevas_hipotesis = []
            # Procesar cada hipótesis actual en el beam
            for secuencia, puntuacion in beam:
                # Si la secuencia ya terminó, mantenerla sin cambios
                if secuencia[-1] == idx_fin:
                    nuevas_hipotesis.append((secuencia, puntuacion))
                    continue
                # Preparar entrada para el decoder
                decoder_input = np.array([secuencia])
                # Obtener predicciones del modelo
                predicciones = self.modelo.predict([texto_procesado, decoder_input], verbose=0)
                # Extraer probabilidades del último token generado
                probabilidades = predicciones[0, -1, :]
                # Seleccionar los k tokens más probables
                top_k_indices = np.argsort(probabilidades)[-ancho_beam:]
                # Crear nuevas hipótesis expandiendo con cada token candidato
                for idx in top_k_indices:
                    nueva_secuencia = secuencia + [idx]
                    # Calcular puntuación acumulada usando log-probabilidad
                    nueva_puntuacion = puntuacion + np.log(probabilidades[idx] + 1e-10)
                    nuevas_hipotesis.append((nueva_secuencia, nueva_puntuacion))
            # Mantener solo las mejores k hipótesis ordenadas por puntuación
            beam = sorted(nuevas_hipotesis, key=lambda x: x[1], reverse=True)[:ancho_beam]
            # Detener si todas las hipótesis han terminado
            if all(seq[-1] == idx_fin for seq, _ in beam):
                break
        
        # Seleccionar la mejor hipótesis del beam final
        mejor_secuencia, _ = beam[0]
        # Convertir secuencia de índices a texto legible
        resumen = self.tokenizador_resumen.secuencia_a_texto(mejor_secuencia, eliminar_tokens_especiales=True)
        return resumen
    
    def generar_resumen(self, texto: str, estrategia: str = config.ESTRATEGIA_GENERACION, **kwargs) -> str:
        """
        Genera un resumen usando la estrategia especificada
        
        Args:
            texto: Texto original a resumir
            estrategia: Estrategia de generación ('greedy' o 'beam_search')
            **kwargs: Argumentos adicionales para la estrategia seleccionada
            
        Returns:
            Resumen generado como cadena de texto
            
        Raises:
            ValueError: Si la estrategia no es reconocida
        """
        # Seleccionar método de generación según estrategia
        if estrategia == 'greedy':
            return self.generar_resumen_greedy(texto, **kwargs)
        elif estrategia == 'beam_search':
            return self.generar_resumen_beam_search(texto, **kwargs)
        else:
            raise ValueError(f"Estrategia no reconocida: {estrategia}")
    
    def generar_resumenes_batch(self, textos: List[str], estrategia: str = 'greedy', mostrar_progreso: bool = True) -> List[str]:
        """
        Genera resúmenes para múltiples textos en lote
        
        Args:
            textos: Lista de textos a resumir
            estrategia: Estrategia de generación a usar
            mostrar_progreso: Si se debe mostrar el progreso en consola
            
        Returns:
            Lista de resúmenes generados
        """
        resumenes = []
        # Procesar cada texto de la lista
        for i, texto in enumerate(textos):
            # Mostrar progreso si está habilitado
            if mostrar_progreso:
                print(f"Generando resumen {i+1}/{len(textos)}...", end='\r')
            # Generar resumen usando la estrategia especificada
            resumen = self.generar_resumen(texto, estrategia=estrategia)
            resumenes.append(resumen)
        # Mostrar mensaje de finalización
        if mostrar_progreso:
            print(f"\n{len(resumenes)} resumenes generados")
        return resumenes

class AnalizadorResumenes:
    """Clase para analizar y comparar textos originales con sus resúmenes"""
    
    @staticmethod
    def calcular_longitud_compresion(texto_original: str, resumen: str) -> float:
        """
        Calcula la tasa de compresión entre texto original y resumen
        
        Args:
            texto_original: Texto completo original
            resumen: Resumen generado
            
        Returns:
            Ratio de palabras (resumen/original)
        """
        # Contar palabras en texto original
        palabras_original = len(texto_original.split())
        # Contar palabras en resumen
        palabras_resumen = len(resumen.split())
        # Calcular y retornar ratio, evitando división por cero
        return palabras_resumen / palabras_original if palabras_original > 0 else 0.0
    
    @staticmethod
    def mostrar_comparacion(texto_original: str, resumen: str):
        """
        Muestra una comparación formateada entre texto original y resumen
        
        Args:
            texto_original: Texto completo original
            resumen: Resumen generado
        """
        # Mostrar texto original con separador visual
        print("\n" + "="*80 + "\nTEXTO ORIGINAL\n" + "="*80)
        print(texto_original)
        # Mostrar conteo de palabras del original
        print(f"\nPalabras: {len(texto_original.split())}")
        # Mostrar resumen generado con separador visual
        print("\n" + "="*80 + "\nRESUMEN GENERADO\n" + "="*80)
        print(resumen)
        # Mostrar conteo de palabras del resumen
        print(f"\nPalabras: {len(resumen.split())}")
        # Calcular y mostrar tasa de compresión
        tasa_compresion = AnalizadorResumenes.calcular_longitud_compresion(texto_original, resumen)
        print(f"Tasa de compresion: {tasa_compresion:.1%}")
        print("="*80 + "\n")

def ejemplo_uso():
    """Función de demostración del generador de resúmenes"""
    # Mostrar encabezado del programa
    print("\n" + "="*80 + "\nGENERADOR DE RESUMENES AUTOMATICOS\n" + "="*80 + "\n")
    # Verificar que existe el modelo entrenado
    if not os.path.exists(config.RUTA_MODELO_ABSTRACTIVO):
        print("No se encontro el modelo entrenado.\n   Por favor, ejecuta primero: python src/entrenamiento.py")
        return
    
    # Inicializar el generador de resúmenes
    generador = GeneradorResumenes()
    # Definir texto de ejemplo para demostración
    texto_ejemplo = """
    La inteligencia artificial ha experimentado un crecimiento exponencial en las últimas décadas,
    transformando numerosos aspectos de nuestra vida cotidiana. Desde asistentes virtuales en nuestros
    teléfonos hasta sistemas de recomendación en plataformas de streaming, la IA está presente en
    múltiples aplicaciones. El aprendizaje profundo, una rama de la IA que utiliza redes neuronales
    artificiales con múltiples capas, ha sido particularmente revolucionario. Estas redes pueden
    aprender patrones complejos en grandes cantidades de datos, permitiendo avances significativos
    en áreas como el reconocimiento de imágenes, el procesamiento del lenguaje natural y la conducción
    autónoma. Sin embargo, el desarrollo de la IA también plantea importantes desafíos éticos y
    sociales que debemos abordar cuidadosamente.
    """
    
    # Generar resumen usando estrategia greedy
    print("Generando resumen (estrategia: greedy)...")
    resumen_greedy = generador.generar_resumen(texto_ejemplo, estrategia='greedy')
    # Mostrar comparación entre texto original y resumen greedy
    AnalizadorResumenes.mostrar_comparacion(texto_ejemplo, resumen_greedy)
    
    # Generar resumen usando estrategia beam search
    print("Generando resumen (estrategia: beam_search)...")
    resumen_beam = generador.generar_resumen(texto_ejemplo, estrategia='beam_search')
    # Mostrar comparación entre texto original y resumen beam search
    AnalizadorResumenes.mostrar_comparacion(texto_ejemplo, resumen_beam)

# Ejecutar función de ejemplo si el script se ejecuta directamente
if __name__ == '__main__':
    ejemplo_uso()
