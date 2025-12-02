"""Módulo de Utilidades - Visualización, métricas y análisis"""
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from typing import List, Dict
import config

def configurar_estilo_graficas():
    plt.style.use(config.ESTILO_GRAFICAS)
    sns.set_palette("husl")

def graficar_longitudes_textos(textos: List[str], titulo: str = "Distribución de Longitudes"):
    configurar_estilo_graficas()
    longitudes = [len(texto.split()) for texto in textos]
    fig, axes = plt.subplots(1, 2, figsize=config.TAMANIO_FIGURA)
    
    axes[0].hist(longitudes, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'{titulo} - Histograma')
    axes[0].set_xlabel('Número de palabras')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(longitudes, vert=True)
    axes[1].set_title(f'{titulo} - Boxplot')
    axes[1].set_ylabel('Número de palabras')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Estadísticas de {titulo}:")
    print(f"   - Promedio: {np.mean(longitudes):.1f} palabras")
    print(f"   - Mediana: {np.median(longitudes):.1f} palabras")
    print(f"   - Mínimo: {np.min(longitudes)} palabras")
    print(f"   - Máximo: {np.max(longitudes)} palabras")
    print(f"   - Desviación estándar: {np.std(longitudes):.1f} palabras\n")

def graficar_distribucion_vocabulario(frecuencias: Dict[str, int], top_n: int = 20):
    configurar_estilo_graficas()
    palabras_ordenadas = sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)
    top_palabras = palabras_ordenadas[:top_n]
    palabras = [p[0] for p in top_palabras]
    frecuencias_top = [p[1] for p in top_palabras]
    
    plt.figure(figsize=config.TAMANIO_FIGURA)
    plt.barh(palabras, frecuencias_top, color='skyblue', edgecolor='black')
    plt.xlabel('Frecuencia')
    plt.title(f'Top {top_n} Palabras Más Frecuentes')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

def graficar_matriz_atencion(pesos_atencion: np.ndarray, palabras_entrada: List[str], palabras_salida: List[str]):
    configurar_estilo_graficas()
    plt.figure(figsize=(12, 8))
    sns.heatmap(pesos_atencion, xticklabels=palabras_entrada, yticklabels=palabras_salida,
                cmap='YlOrRd', cbar_kws={'label': 'Peso de Atención'})
    plt.xlabel('Texto de Entrada')
    plt.ylabel('Resumen Generado')
    plt.title('Matriz de Atención')
    plt.tight_layout()
    plt.show()

def calcular_metricas_rouge_simple(referencia: str, generado: str) -> Dict[str, float]:
    palabras_ref = set(referencia.lower().split())
    palabras_gen = set(generado.lower().split())
    interseccion = palabras_ref.intersection(palabras_gen)
    
    if len(palabras_ref) == 0:
        precision, recall = 0.0, 0.0
    else:
        precision = len(interseccion) / len(palabras_gen) if len(palabras_gen) > 0 else 0.0
        recall = len(interseccion) / len(palabras_ref)
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'rouge-1-precision': precision, 'rouge-1-recall': recall, 'rouge-1-f1': f1}

def mostrar_ejemplo_resumen(texto: str, resumen_referencia: str, resumen_generado: str):
    print("\n" + "="*80 + "\n📄 TEXTO ORIGINAL\n" + "="*80)
    print(texto)
    print(f"\n📊 Longitud: {len(texto.split())} palabras")
    print("\n" + "="*80 + "\n✅ RESUMEN DE REFERENCIA\n" + "="*80)
    print(resumen_referencia)
    print(f"\n📊 Longitud: {len(resumen_referencia.split())} palabras")
    print("\n" + "="*80 + "\n🤖 RESUMEN GENERADO\n" + "="*80)
    print(resumen_generado)
    print(f"\n📊 Longitud: {len(resumen_generado.split())} palabras")
    
    metricas = calcular_metricas_rouge_simple(resumen_referencia, resumen_generado)
    print("\n" + "="*80 + "\n📈 MÉTRICAS ROUGE-1\n" + "="*80)
    print(f"   Precisión: {metricas['rouge-1-precision']:.3f}")
    print(f"   Recall:    {metricas['rouge-1-recall']:.3f}")
    print(f"   F1-Score:  {metricas['rouge-1-f1']:.3f}")
    print("="*80 + "\n")

def crear_reporte_entrenamiento(historia, nombre_archivo: str = "reporte_entrenamiento.txt"):
    import os
    ruta_reporte = os.path.join(config.RUTA_GRAFICAS, nombre_archivo)
    
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE ENTRENAMIENTO - SISTEMA DE RESUMEN AUTOMÁTICO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Número de épocas: {len(historia.history['loss'])}\n\n")
        f.write("MÉTRICAS FINALES:\n" + "-"*80 + "\n")
        f.write(f"Loss (Entrenamiento):     {historia.history['loss'][-1]:.4f}\n")
        f.write(f"Loss (Validación):        {historia.history['val_loss'][-1]:.4f}\n")
        f.write(f"Accuracy (Entrenamiento): {historia.history['accuracy'][-1]:.4f}\n")
        f.write(f"Accuracy (Validación):    {historia.history['val_accuracy'][-1]:.4f}\n\n")
        f.write("MEJORES MÉTRICAS:\n" + "-"*80 + "\n")
        mejor_val_loss = min(historia.history['val_loss'])
        mejor_val_acc = max(historia.history['val_accuracy'])
        f.write(f"Mejor Loss (Validación):     {mejor_val_loss:.4f}\n")
        f.write(f"Mejor Accuracy (Validación): {mejor_val_acc:.4f}\n\n")
        f.write("="*80 + "\n")
    print(f"📄 Reporte guardado en: {ruta_reporte}")

def imprimir_banner(texto: str):
    longitud = len(texto) + 4
    print("\n" + "="*longitud + f"\n  {texto}\n" + "="*longitud + "\n")

if __name__ == '__main__':
    print("=== Ejemplos de Utilidades ===\n")
    textos_ejemplo = ["Este es un texto corto", "Este es un texto un poco más largo con más palabras",
                      "Y este es un texto considerablemente más extenso que los anteriores con muchas más palabras para analizar"]
    graficar_longitudes_textos(textos_ejemplo, "Textos de Ejemplo")
    
    referencia = "el gato está en el tejado"
    generado = "el gato está arriba"
    metricas = calcular_metricas_rouge_simple(referencia, generado)
    print(f"Métricas ROUGE: {metricas}")