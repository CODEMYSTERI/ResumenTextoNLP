"""
Módulo de Preprocesamiento de Texto
Contiene funciones para limpiar, tokenizar y preparar textos para el modelo
"""

import re
import string
import unicodedata
import numpy as np
from typing import List, Tuple, Dict
import pickle


class LimpiadorTexto:
    """
    Clase para limpiar y normalizar textos en español
    """
    
    def __init__(self, mantener_acentos: bool = True, minusculas: bool = True):
        """
        Inicializa el limpiador de texto
        
        Args:
            mantener_acentos: Si mantener los acentos en el texto
            minusculas: Si convertir todo a minúsculas
        """
        self.mantener_acentos = mantener_acentos
        self.minusculas = minusculas
        self.puntuacion = string.punctuation
        
    def limpiar_texto(self, texto: str) -> str:
        """
        Limpia un texto eliminando elementos no deseados
        
        Args:
            texto: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        if not texto or not isinstance(texto, str):
            return ""
        
        # Convertir a minúsculas si es necesario
        if self.minusculas:
            texto = texto.lower()
        
        # Eliminar URLs
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
        
        # Eliminar menciones y hashtags (si existen)
        texto = re.sub(r'@\w+|#\w+', '', texto)
        
        # Eliminar emails
        texto = re.sub(r'\S+@\S+', '', texto)
        
        # Eliminar números (opcional - comentar si quieres mantener números)
        # texto = re.sub(r'\d+', '', texto)
        
        # Normalizar espacios en blanco
        texto = re.sub(r'\s+', ' ', texto)
        
        # Eliminar acentos si es necesario
        if not self.mantener_acentos:
            texto = self._eliminar_acentos(texto)
        
        # Eliminar espacios al inicio y final
        texto = texto.strip()
        
        return texto
    
    def _eliminar_acentos(self, texto: str) -> str:
        """Elimina acentos del texto"""
        texto_normalizado = unicodedata.normalize('NFD', texto)
        return ''.join(char for char in texto_normalizado 
                      if unicodedata.category(char) != 'Mn')
    
    def separar_oraciones(self, texto: str) -> List[str]:
        """
        Separa un texto en oraciones
        
        Args:
            texto: Texto a separar
            
        Returns:
            Lista de oraciones
        """
        # Patrones para separar oraciones en español
        patrones = r'[.!?]+\s+'
        oraciones = re.split(patrones, texto)
        
        # Limpiar oraciones vacías
        oraciones = [oracion.strip() for oracion in oraciones if oracion.strip()]
        
        return oraciones


class TokenizadorTexto:
    """
    Clase para tokenizar textos y convertirlos a secuencias numéricas
    """
    
    def __init__(self, 
                 tamanio_vocabulario: int = 50000,
                 palabra_desconocida: str = '<UNK>',
                 palabra_inicio: str = '<START>',
                 palabra_fin: str = '<END>',
                 palabra_relleno: str = '<PAD>'):
        """
        Inicializa el tokenizador
        
        Args:
            tamanio_vocabulario: Tamaño máximo del vocabulario
            palabra_desconocida: Token para palabras desconocidas
            palabra_inicio: Token de inicio de secuencia
            palabra_fin: Token de fin de secuencia
            palabra_relleno: Token de relleno (padding)
        """
        self.tamanio_vocabulario = tamanio_vocabulario
        self.palabra_desconocida = palabra_desconocida
        self.palabra_inicio = palabra_inicio
        self.palabra_fin = palabra_fin
        self.palabra_relleno = palabra_relleno
        
        # Diccionarios palabra <-> índice
        self.palabra_a_indice = {}
        self.indice_a_palabra = {}
        
        # Contador de frecuencias
        self.frecuencias_palabras = {}
        
        # Indicadores
        self.ajustado = False
        
    def ajustar(self, textos: List[str]):
        """
        Construye el vocabulario a partir de una lista de textos
        
        Args:
            textos: Lista de textos para construir el vocabulario
        """
        # Contar frecuencias de palabras
        for texto in textos:
            palabras = texto.split()
            for palabra in palabras:
                self.frecuencias_palabras[palabra] = \
                    self.frecuencias_palabras.get(palabra, 0) + 1
        
        # Ordenar palabras por frecuencia
        palabras_ordenadas = sorted(
            self.frecuencias_palabras.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Crear vocabulario con las palabras más frecuentes
        # Reservar índices para tokens especiales
        tokens_especiales = [
            self.palabra_relleno,
            self.palabra_desconocida,
            self.palabra_inicio,
            self.palabra_fin
        ]
        
        # Agregar tokens especiales
        for idx, token in enumerate(tokens_especiales):
            self.palabra_a_indice[token] = idx
            self.indice_a_palabra[idx] = token
        
        # Agregar palabras del vocabulario
        idx_actual = len(tokens_especiales)
        for palabra, _ in palabras_ordenadas[:self.tamanio_vocabulario - len(tokens_especiales)]:
            self.palabra_a_indice[palabra] = idx_actual
            self.indice_a_palabra[idx_actual] = palabra
            idx_actual += 1
        
        self.ajustado = True
        print(f"Vocabulario construido con {len(self.palabra_a_indice)} palabras")
    
    def texto_a_secuencia(self, texto: str, agregar_tokens_especiales: bool = False) -> List[int]:
        """
        Convierte un texto a una secuencia de índices
        
        Args:
            texto: Texto a convertir
            agregar_tokens_especiales: Si agregar tokens de inicio y fin
            
        Returns:
            Lista de índices
        """
        if not self.ajustado:
            raise ValueError("El tokenizador debe ser ajustado primero con ajustar()")
        
        palabras = texto.split()
        indice_desconocido = self.palabra_a_indice[self.palabra_desconocida]
        
        secuencia = [
            self.palabra_a_indice.get(palabra, indice_desconocido)
            for palabra in palabras
        ]
        
        if agregar_tokens_especiales:
            indice_inicio = self.palabra_a_indice[self.palabra_inicio]
            indice_fin = self.palabra_a_indice[self.palabra_fin]
            secuencia = [indice_inicio] + secuencia + [indice_fin]
        
        return secuencia
    
    def secuencia_a_texto(self, secuencia: List[int], eliminar_tokens_especiales: bool = True) -> str:
        """
        Convierte una secuencia de índices a texto
        
        Args:
            secuencia: Lista de índices
            eliminar_tokens_especiales: Si eliminar tokens especiales
            
        Returns:
            Texto reconstruido
        """
        tokens_especiales = {
            self.palabra_a_indice.get(self.palabra_relleno, -1),
            self.palabra_a_indice.get(self.palabra_inicio, -1),
            self.palabra_a_indice.get(self.palabra_fin, -1),
        }
        
        palabras = []
        for idx in secuencia:
            if eliminar_tokens_especiales and idx in tokens_especiales:
                continue
            palabra = self.indice_a_palabra.get(idx, self.palabra_desconocida)
            palabras.append(palabra)
        
        return ' '.join(palabras)
    
    def rellenar_secuencias(self, 
                           secuencias: List[List[int]], 
                           longitud_maxima: int,
                           relleno_posterior: bool = True) -> np.ndarray:
        
        indice_relleno = self.palabra_a_indice[self.palabra_relleno]
        
        secuencias_rellenadas = np.full(
            (len(secuencias), longitud_maxima),
            indice_relleno,
            dtype=np.int32
        )
        
        for i, secuencia in enumerate(secuencias):
            if len(secuencia) == 0:
                continue
            
            # Truncar si es necesario
            secuencia = secuencia[:longitud_maxima]
            
            if relleno_posterior:
                secuencias_rellenadas[i, :len(secuencia)] = secuencia
            else:
                secuencias_rellenadas[i, -len(secuencia):] = secuencia
        
        return secuencias_rellenadas
    
    def guardar(self, ruta: str):
        """Guarda el tokenizador en un archivo"""
        with open(ruta, 'wb') as archivo:
            pickle.dump(self, archivo)
        print(f"Tokenizador guardado en {ruta}")
    
    @staticmethod
    def cargar(ruta: str) -> 'TokenizadorTexto':
        """Carga un tokenizador desde un archivo"""
        with open(ruta, 'rb') as archivo:
            tokenizador = pickle.load(archivo)
        print(f"Tokenizador cargado desde {ruta}")
        return tokenizador


class PreparadorDatos:
    """
    Clase para preparar datos de entrenamiento para el modelo
    """
    
    def __init__(self, 
                 limpiador: LimpiadorTexto,
                 tokenizador_texto: TokenizadorTexto,
                 tokenizador_resumen: TokenizadorTexto):
        """
        Inicializa el preparador de datos
        
        Args:
            limpiador: Instancia de LimpiadorTexto
            tokenizador_texto: Tokenizador para textos originales
            tokenizador_resumen: Tokenizador para resúmenes
        """
        self.limpiador = limpiador
        self.tokenizador_texto = tokenizador_texto
        self.tokenizador_resumen = tokenizador_resumen
    
    def preparar_datos(self,
                      textos: List[str],
                      resumenes: List[str],
                      longitud_maxima_texto: int,
                      longitud_maxima_resumen: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento
        
        Args:
            textos: Lista de textos originales
            resumenes: Lista de resúmenes
            longitud_maxima_texto: Longitud máxima para textos
            longitud_maxima_resumen: Longitud máxima para resúmenes
            
        Returns:
            Tupla (textos_procesados, resumenes_entrada, resumenes_salida)
        """
        # Limpiar textos
        textos_limpios = [self.limpiador.limpiar_texto(texto) for texto in textos]
        resumenes_limpios = [self.limpiador.limpiar_texto(resumen) for resumen in resumenes]
        
        # Convertir a secuencias
        secuencias_texto = [
            self.tokenizador_texto.texto_a_secuencia(texto)
            for texto in textos_limpios
        ]
        
        secuencias_resumen = [
            self.tokenizador_resumen.texto_a_secuencia(resumen, agregar_tokens_especiales=True)
            for resumen in resumenes_limpios
        ]
        
        # Rellenar secuencias
        textos_rellenados = self.tokenizador_texto.rellenar_secuencias(
            secuencias_texto, longitud_maxima_texto
        )
        
        resumenes_rellenados = self.tokenizador_resumen.rellenar_secuencias(
            secuencias_resumen, longitud_maxima_resumen
        )
        
        # Para el decoder: entrada y salida (shifted)
        resumenes_entrada = resumenes_rellenados[:, :-1]
        resumenes_salida = resumenes_rellenados[:, 1:]
        
        return textos_rellenados, resumenes_entrada, resumenes_salida


# Funciones auxiliares
def calcular_estadisticas_texto(textos: List[str]) -> Dict:
    """
    Calcula estadísticas sobre una lista de textos
    
    Args:
        textos: Lista de textos
        
    Returns:
        Diccionario con estadísticas
    """
    longitudes = [len(texto.split()) for texto in textos]
    
    return {
        'numero_textos': len(textos),
        'longitud_promedio': np.mean(longitudes),
        'longitud_mediana': np.median(longitudes),
        'longitud_minima': np.min(longitudes),
        'longitud_maxima': np.max(longitudes),
        'desviacion_estandar': np.std(longitudes)
    }


if __name__ == '__main__':
    # Ejemplo de uso
    print("=== Ejemplo de Preprocesamiento ===\n")
    
    # Crear limpiador
    limpiador = LimpiadorTexto(mantener_acentos=True, minusculas=True)
    
    # Texto de ejemplo
    texto_ejemplo = """
    ¡Hola! Este es un EJEMPLO de texto con URLs https://ejemplo.com 
    y algunos números 123. También tiene @menciones y #hashtags.
    """
    
    texto_limpio = limpiador.limpiar_texto(texto_ejemplo)
    print(f"Texto original: {texto_ejemplo}")
    print(f"Texto limpio: {texto_limpio}\n")
    
    # Crear tokenizador
    tokenizador = TokenizadorTexto(tamanio_vocabulario=100)
    
    # Textos de ejemplo
    textos = [
        "el gato está en el tejado",
        "el perro corre por el parque",
        "el gato y el perro son amigos"
    ]
    
    # Ajustar tokenizador
    tokenizador.ajustar(textos)
    
    # Convertir a secuencias
    secuencia = tokenizador.texto_a_secuencia(textos[0])
    print(f"Texto: {textos[0]}")
    print(f"Secuencia: {secuencia}")
    
    # Convertir de vuelta a texto
    texto_reconstruido = tokenizador.secuencia_a_texto(secuencia)
    print(f"Texto reconstruido: {texto_reconstruido}\n")
    
    # Rellenar secuencias
    secuencias = [tokenizador.texto_a_secuencia(t) for t in textos]
    secuencias_rellenadas = tokenizador.rellenar_secuencias(secuencias, longitud_maxima=10)
    print(f"Secuencias rellenadas:\n{secuencias_rellenadas}")
