"""Módulo de Predicción y Generación de Resúmenes"""
import os, sys, numpy as np, tensorflow as tf
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocesamiento import LimpiadorTexto, TokenizadorTexto

class GeneradorResumenes:
    def __init__(self, ruta_modelo: str = config.RUTA_MODELO_ABSTRACTIVO,
                 ruta_tokenizador_texto: str = config.RUTA_TOKENIZADOR_TEXTO,
                 ruta_tokenizador_resumen: str = config.RUTA_TOKENIZADOR_RESUMEN):
        self.limpiador = LimpiadorTexto(mantener_acentos=True, minusculas=True)
        print("📂 Cargando tokenizadores...")
        self.tokenizador_texto = TokenizadorTexto.cargar(ruta_tokenizador_texto)
        self.tokenizador_resumen = TokenizadorTexto.cargar(ruta_tokenizador_resumen)
        
        print("🤖 Cargando modelo...")
        from src.modelo_abstractivo import Encoder, Decoder, MecanismoAtencionBahdanau
        self.modelo = tf.keras.models.load_model(ruta_modelo, custom_objects={
            'Encoder': Encoder, 'Decoder': Decoder, 'MecanismoAtencionBahdanau': MecanismoAtencionBahdanau}, compile=False)
        print("✅ Generador de resúmenes listo!\n")
    
    def preprocesar_texto(self, texto: str) -> np.ndarray:
        texto_limpio = self.limpiador.limpiar_texto(texto)
        secuencia = self.tokenizador_texto.texto_a_secuencia(texto_limpio)
        secuencia_rellenada = self.tokenizador_texto.rellenar_secuencias([secuencia], config.LONGITUD_MAXIMA_TEXTO)
        return secuencia_rellenada
    
    def generar_resumen_greedy(self, texto: str, longitud_maxima: int = None) -> str:
        if longitud_maxima is None:
            longitud_maxima = config.LONGITUD_MAXIMA_RESUMEN
        texto_procesado = self.preprocesar_texto(texto)
        idx_inicio = self.tokenizador_resumen.palabra_a_indice[config.PALABRA_INICIO]
        idx_fin = self.tokenizador_resumen.palabra_a_indice[config.PALABRA_FIN]
        secuencia_decoder = [idx_inicio]
        
        for _ in range(longitud_maxima):
            decoder_input = np.array([secuencia_decoder])
            predicciones = self.modelo.predict([texto_procesado, decoder_input], verbose=0)
            siguiente_token = np.argmax(predicciones[0, -1, :])
            if siguiente_token == idx_fin:
                break
            secuencia_decoder.append(siguiente_token)
        
        resumen = self.tokenizador_resumen.secuencia_a_texto(secuencia_decoder, eliminar_tokens_especiales=True)
        return resumen
    
    def generar_resumen_beam_search(self, texto: str, ancho_beam: int = config.ANCHO_BEAM, longitud_maxima: int = None) -> str:
        if longitud_maxima is None:
            longitud_maxima = config.LONGITUD_MAXIMA_RESUMEN
        texto_procesado = self.preprocesar_texto(texto)
        idx_inicio = self.tokenizador_resumen.palabra_a_indice[config.PALABRA_INICIO]
        idx_fin = self.tokenizador_resumen.palabra_a_indice[config.PALABRA_FIN]
        beam = [([idx_inicio], 0.0)]
        
        for _ in range(longitud_maxima):
            nuevas_hipotesis = []
            for secuencia, puntuacion in beam:
                if secuencia[-1] == idx_fin:
                    nuevas_hipotesis.append((secuencia, puntuacion))
                    continue
                decoder_input = np.array([secuencia])
                predicciones = self.modelo.predict([texto_procesado, decoder_input], verbose=0)
                probabilidades = predicciones[0, -1, :]
                top_k_indices = np.argsort(probabilidades)[-ancho_beam:]
                for idx in top_k_indices:
                    nueva_secuencia = secuencia + [idx]
                    nueva_puntuacion = puntuacion + np.log(probabilidades[idx] + 1e-10)
                    nuevas_hipotesis.append((nueva_secuencia, nueva_puntuacion))
            beam = sorted(nuevas_hipotesis, key=lambda x: x[1], reverse=True)[:ancho_beam]
            if all(seq[-1] == idx_fin for seq, _ in beam):
                break
        
        mejor_secuencia, _ = beam[0]
        resumen = self.tokenizador_resumen.secuencia_a_texto(mejor_secuencia, eliminar_tokens_especiales=True)
        return resumen
    
    def generar_resumen(self, texto: str, estrategia: str = config.ESTRATEGIA_GENERACION, **kwargs) -> str:
        if estrategia == 'greedy':
            return self.generar_resumen_greedy(texto, **kwargs)
        elif estrategia == 'beam_search':
            return self.generar_resumen_beam_search(texto, **kwargs)
        else:
            raise ValueError(f"Estrategia no reconocida: {estrategia}")
    
    def generar_resumenes_batch(self, textos: List[str], estrategia: str = 'greedy', mostrar_progreso: bool = True) -> List[str]:
        resumenes = []
        for i, texto in enumerate(textos):
            if mostrar_progreso:
                print(f"Generando resumen {i+1}/{len(textos)}...", end='\r')
            resumen = self.generar_resumen(texto, estrategia=estrategia)
            resumenes.append(resumen)
        if mostrar_progreso:
            print(f"\n✅ {len(resumenes)} resúmenes generados")
        return resumenes

class AnalizadorResumenes:
    @staticmethod
    def calcular_longitud_compresion(texto_original: str, resumen: str) -> float:
        palabras_original = len(texto_original.split())
        palabras_resumen = len(resumen.split())
        return palabras_resumen / palabras_original if palabras_original > 0 else 0.0
    
    @staticmethod
    def mostrar_comparacion(texto_original: str, resumen: str):
        print("\n" + "="*80 + "\n📄 TEXTO ORIGINAL\n" + "="*80)
        print(texto_original)
        print(f"\n📊 Palabras: {len(texto_original.split())}")
        print("\n" + "="*80 + "\n📝 RESUMEN GENERADO\n" + "="*80)
        print(resumen)
        print(f"\n📊 Palabras: {len(resumen.split())}")
        tasa_compresion = AnalizadorResumenes.calcular_longitud_compresion(texto_original, resumen)
        print(f"📉 Tasa de compresión: {tasa_compresion:.1%}")
        print("="*80 + "\n")

def ejemplo_uso():
    print("\n" + "="*80 + "\n🚀 GENERADOR DE RESÚMENES AUTOMÁTICOS\n" + "="*80 + "\n")
    if not os.path.exists(config.RUTA_MODELO_ABSTRACTIVO):
        print("❌ No se encontró el modelo entrenado.\n   Por favor, ejecuta primero: python src/entrenamiento.py")
        return
    
    generador = GeneradorResumenes()
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
    
    print("🔍 Generando resumen (estrategia: greedy)...")
    resumen_greedy = generador.generar_resumen(texto_ejemplo, estrategia='greedy')
    AnalizadorResumenes.mostrar_comparacion(texto_ejemplo, resumen_greedy)
    
    print("🔍 Generando resumen (estrategia: beam_search)...")
    resumen_beam = generador.generar_resumen(texto_ejemplo, estrategia='beam_search')
    AnalizadorResumenes.mostrar_comparacion(texto_ejemplo, resumen_beam)

if __name__ == '__main__':
    ejemplo_uso()
