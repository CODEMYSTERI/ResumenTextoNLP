"""RESUMEN.PY - Resumidor Automático - Pega tu texto y obtén un resumen"""
import sys, os, numpy as np, pandas as pd, re, math
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

class ResumidorInteligente:
    def __init__(self):
        self.stopwords = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
            'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
            'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
            'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'muy', 'sin', 'vez',
            'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo', 'yo', 'también',
            'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero', 'desde', 'grande',
            'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella', 'sí', 'día', 'uno',
            'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa', 'tanto', 'hombre',
            'parecer', 'nuestro', 'tan', 'donde', 'ahora', 'parte', 'después', 'vida',
            'quedar', 'siempre', 'creer', 'hablar', 'llevar', 'dejar', 'nada', 'cada',
            'seguir', 'menos', 'nuevo', 'encontrar', 'algo', 'solo', 'estos', 'trabajar',
            'ha', 'han', 'he', 'has', 'son', 'es', 'fue', 'sido', 'está', 'están', 'esta',
            'las', 'los', 'del', 'al', 'una'
        }
    
    def limpiar_texto(self, texto):
        texto = texto.lower()
        texto = re.sub(r'http\S+|www\S+', '', texto)
        texto = re.sub(r'@\w+|#\w+', '', texto)
        return texto.strip()
    
    def separar_oraciones(self, texto):
        lineas = [linea.strip() for linea in texto.split('\n') if linea.strip()]
        if len(lineas) <= 3:
            oraciones = re.split(r'[.!?]+', texto)
            oraciones = [o.strip() for o in oraciones if o.strip() and len(o.split()) > 3]
            if len(oraciones) > len(lineas):
                return oraciones
        return lineas
    
    def calcular_importancia(self, fragmento, idf_global, palabras_importantes):
        palabras = fragmento.lower().split()
        palabras_filtradas = [p for p in palabras if p and len(p) > 2 and p not in self.stopwords]
        if not palabras_filtradas:
            return 0.0
        
        freq = Counter(palabras_filtradas)
        score_tfidf, palabras_unicas = 0.0, 0
        for palabra, count in freq.items():
            tf = count / len(palabras_filtradas)
            idf = idf_global.get(palabra, 0)
            score_tfidf += tf * idf
            if palabra in palabras_importantes:
                palabras_unicas += 1
        
        score_tfidf = score_tfidf / len(palabras_filtradas)
        score_unicas = palabras_unicas / len(palabras_filtradas)
        longitud_ideal = 10
        penalizacion = max(0.5, 1.0 - abs(len(palabras) - longitud_ideal) / 20)
        return (score_tfidf * 0.7 + score_unicas * 0.3) * penalizacion
    
    def generar_resumen(self, texto):
        texto_limpio = self.limpiar_texto(texto)
        fragmentos = self.separar_oraciones(texto_limpio)
        
        if len(fragmentos) == 0:
            return ""
        if len(fragmentos) <= 2:
            return '. '.join(fragmentos) + '.'
        
        todas_palabras = set()
        fragmentos_palabras = []
        for fragmento in fragmentos:
            palabras = [p for p in fragmento.lower().split() if p and len(p) > 2]
            fragmentos_palabras.append(palabras)
            todas_palabras.update(palabras)
        
        idf = {}
        for palabra in todas_palabras:
            docs_con_palabra = sum(1 for palabras in fragmentos_palabras if palabra in palabras)
            idf[palabra] = math.log(len(fragmentos) / (1 + docs_con_palabra))
        
        todas_palabras_freq = Counter()
        for palabras in fragmentos_palabras:
            todas_palabras_freq.update([p for p in palabras if p not in self.stopwords])
        
        num_top = max(5, len(todas_palabras_freq) // 5)
        palabras_importantes = set([palabra for palabra, _ in todas_palabras_freq.most_common(num_top)])
        
        importancias = [self.calcular_importancia(fragmento, idf, palabras_importantes) for fragmento in fragmentos]
        
        num_fragmentos_total = len(fragmentos)
        if num_fragmentos_total <= 5:
            num_seleccionar = max(2, num_fragmentos_total - 1)
        elif num_fragmentos_total <= 10:
            num_seleccionar = max(3, int(num_fragmentos_total * 0.4))
        else:
            num_seleccionar = max(4, int(num_fragmentos_total * 0.35))
        
        indices_ordenados = np.argsort(importancias)[::-1]
        indices_seleccionados = sorted(indices_ordenados[:num_seleccionar])
        resumen_fragmentos = [fragmentos[i] for i in indices_seleccionados]
        resumen = '. '.join(resumen_fragmentos)
        if not resumen.endswith('.'):
            resumen += '.'
        return resumen

def guardar_ejemplo(texto, resumen):
    try:
        ruta_datos = config.RUTA_DATOS_ENTRENAMIENTO
        if os.path.exists(ruta_datos):
            df = pd.read_csv(ruta_datos)
        else:
            df = pd.DataFrame(columns=['texto', 'resumen'])
        nuevo_ejemplo = pd.DataFrame([{'texto': texto, 'resumen': resumen}])
        df = pd.concat([df, nuevo_ejemplo], ignore_index=True)
        os.makedirs(os.path.dirname(ruta_datos), exist_ok=True)
        df.to_csv(ruta_datos, index=False)
        return len(df)
    except Exception as e:
        print(f"\n Error al guardar: {e}")
        return 0

def main():
    print("\n" + "="*80 + "\n" + " "*20 + " RESUMIDOR AUTOMATICO INTELIGENTE\n" + " "*10 + "Resume y guarda automaticamente para entrenar el modelo\n" + "="*80 + "\n")
    resumidor = ResumidorInteligente()
    
    while True:
        print("\n Pega tu texto (presiona Enter para terminar):\n>>> ", end="", flush=True)
        lineas, primera_linea = [], True
        try:
            while True:
                linea = input()
                if not linea.strip():
                    if not primera_linea:
                        break
                else:
                    lineas.append(linea)
                    primera_linea = False
        except EOFError:
            pass
        
        if not lineas:
            print("\n👋 ¡Hasta luego!")
            break
        
        texto = " ".join(lineas).strip()
        if len(texto.split()) < 10:
            print("\n⚠️  El texto es muy corto. Intenta con al menos 10 palabras.")
            continue
        
        print("\n⏳ Generando resumen...")
        resumen = resumidor.generar_resumen(texto)
        
        print("\n" + "="*80 + "\n📝 RESUMEN:\n" + "="*80)
        print(resumen)
        print("\n" + "="*80)
        print(f"📊 Original: {len(texto.split())} palabras | Resumen: {len(resumen.split())} palabras")
        print(f"📉 Compresión: {len(resumen.split())/len(texto.split())*100:.1f}%")
        
        total_ejemplos = guardar_ejemplo(texto, resumen)
        if total_ejemplos > 0:
            print(f"✅ Guardado automáticamente ({total_ejemplos} ejemplos totales)")
        
        print("\n¿Otro texto? (Enter para continuar, 'n' para salir): ", end="")
        if input().lower() == 'n':
            print("\n👋 ¡Hasta luego!")
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programa interrumpido\n👋 ¡Hasta luego!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
