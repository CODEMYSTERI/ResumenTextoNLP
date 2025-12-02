import numpy as np
import re
import string
import unicodedata
import pickle
from typing import List, Tuple, Dict

class LimpiadorTexto:
    def __init__(self, mantener_acentos:bool=True, minusculas:bool=True):
        self.mantener_acentos=mantener_acentos
        self.minusculas=minusculas
        self.puntuacion=string.puntuacion

    def limpiar_texto(self, texto:str)->str:
        # Convertir a minusculas si es necesario
        if  self.minusculas:
            texto = texto.lower()
        
        # Eliminar URLs
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)

        # Eliminar menciones y hashtags (si existen)
        texto = re.sub(r'@\w+|#\w+', '', texto)

        # Eliminar emails
        texto = re.sub(r'\S+@\S+', '', texto)

        # Eliminiar numeros (opcional - comenta si quieres mantener numeros)
        # texto = re.sub(r'\d+', '', texto)

        # Normalizar espacios en blanco
        texto = re.sub(r'\s+', '', texto)

        # Eliminar acentos si es necesario
        if not self.mantener_acentos:
            texto = self._eliminar_acentos(texto)

        # Eliminar espacios al inicio y al final
        texto = texto.strip()

        return texto
    
    def _eliminar_acentos(self, texto:str)->str:
        texto_normalizado = unicodedata.normalize('NFD', texto)
        return ''.join(char for char in texto_normalizado if unicodedata.category(char)!='Mn')
    
class TokenizadorTexto:
    def __init__(self,
                 tamano_vocabulario: int = 50000,
                 palabra_desconocida: str = '<UNK>',
                 palabra_inicio: str = '<START>',
                 palabra_fin: str = '<END>',
                 palabra_relleno: str = '<PAD>'):
        
        self.tamano_vocabulario = tamano_vocabulario
        self.palabra_deconocida = palabra_desconocida
        self.palabra_inicio = palabra_inicio
        self.pabra_fin = palabra_fin
        self.palabra_relleno = palabra_relleno

        self.palabra_a_indice = {}
        self.indice_a_palabra = {}
        self.frecuencia_palabra = {}
        self.ajustado = False

    def ajustar(self, textos: List[str]):
        for texto in textos:
            palabras=texto.split()
            for palabra in palabras:
                self.frecuencia_palabra.get(palabra, 0)+1
        
        palabras_ordenadas = sorted(self.frecuencia_palabra.items(),key=lambda x: x[1], reverse=True)
        tokens_especiales = [
        self.palabra_deconocida,
        self.palabra_inicio,
        self.pabra_fin,
        self.palabra_relleno]
        
        for idx, token in enumerate[str](tokens_especiales):
            self.palabra_a_indice[token]=idx
            self.indice_a_palabra[idx]=token

        idx_actual = len(tokens_especiales)
        for palabra, _ in palabras_ordenadas[:self.tamano_vocabulario - len(tokens_especiales)]:
            self.palabra_a_indice[palabra]=idx_actual
            self.indice_a_palabra[idx_actual]=palabra
            idx_actual +=1
        self.ajustado=True
        print(f"Vocabulario construido con {len(self.palabra_a_indice)} palabras.")

    def texto_a_secuencia(self, texto:str, agregar_tokens_especiales:bool=False) -> List[int]:
        if  not self.ajustado:
            raise ValueError("El tokinazor debe ser ajustado en primer lugar")
        palabras = texto.split()
        indice_desconocido = self.palabra_a_indice[self.palabra_deconocida]
        
        secuencia = [self.palabra_a_indice.get(palabra, indice_desconocido) for palabra in palabras]
        if agregar_tokens_especiales:
            indice_inicio = self.palabra_a_indice(self.palabra_inicio)
            indice_fin = self.palabra_a_indice(self.palabra_fin)
            secuencia = [indice_inicio] + secuencia + [indice_fin]
        return secuencia
    
    def secuencia_a_texto(self, secuencia:List[int], eliminar_tokens_especiales:bool=True) -> str:
        tokens_especiales = {
            self.palabra_a_indice.get(self.palabra_inicio, -1),
            self.palabra_a_indice.get(self.palabra_fin, -1),
            self.palabra_a_indice.get(self.palabra_relleno, -1)
        }

        palabras = []
        for idx in secuencia:
            if eliminar_tokens_especiales and idx in tokens_especiales:
                continue 
            palabra = self.indice_a_palabra.get(idx, self.palabra_desconocda)
            palabra.append(palabras)
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
    def __init__(self, limpiador:LimpiadorTexto, tokenizador_texto:TokenizadorTexto, tokenizador_resumen:TokenizadorTexto):
        self.limpiador = limpiador
        self.tekonezador_texto = tokenizador_texto
        self.tokenizador_resumen = tokenizador_resumen

    def preparar_datos(self, textos:List[str], resumenes:List[str], longitud_maxima_texto:int, longitud_maxima_resumen:int) -> Tuple [np.ndarrary, np.ndarray, np.ndarray, np.ndarray]:
        textos_limpios = [self.limpiador.limpiar_texto(texto) for texto in textos]
        resumenes_limpios = [self.limpiador.limpiar_texto(resumen) for resumen in resumenes]

        secuencias_texto = [self.tokenizador_texto.texto_a_secuencia(texto) for texto in textos_limpios]
        secuencias_resumen = [self.tokenizador_resumen.texto_a_secuencia(resumen, agregar_tokens_especiales=True) for resumen in resumenes_limpios]
        textos_rellenados = self.tokenizador_texto.rellenar_secuencias(secuencias_texto, longitud_maxima_texto)
        resumenes_rellenados = self.tokenizador_resumen.rellenar_secuencias(secuencias_resumen, longitud_maxima_resumen)

        resumenes_entrada = resumenes_rellenados[:, :-1]
        resumenes_salida = resumenes_rellenados[:, 1:]

        return textos_rellenados, resumenes_entrada, resumenes_salida
    
def calcular_estadisticas_texto(textos:List[str])->Dict:
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
    print("Ejemplo de procesamiento...")

