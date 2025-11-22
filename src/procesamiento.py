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
        

