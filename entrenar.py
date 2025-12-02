"""
ENTRENAR.PY - Entrenamiento Interactivo del Modelo
Agrega tus propios textos y resúmenes para entrenar el modelo Seq2Seq
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    print("\n" + "="*80)
    print(" "*15 + "🎓 ENTRENAMIENTO INTERACTIVO DEL MODELO")
    print(" "*20 + "Agrega tus propios textos")
    print("="*80 + "\n")
    
    # Cargar datos existentes si existen
    ruta_datos = config.RUTA_DATOS_ENTRENAMIENTO
    
    if os.path.exists(ruta_datos):
        df_existente = pd.read_csv(ruta_datos)
        print(f"✅ Datos existentes cargados: {len(df_existente)} ejemplos\n")
    else:
        df_existente = pd.DataFrame(columns=['texto', 'resumen'])
        print("📝 No hay datos previos. Empezando desde cero.\n")
    
    nuevos_datos = []
    
    while True:
        print("\n" + "-"*80)
        print("📝 OPCIONES:")
        print("   1. Agregar nuevo ejemplo (texto + resumen)")
        print("   2. Ver ejemplos actuales")
        print("   3. Entrenar modelo con los datos")
        print("   0. Salir")
        print("-"*80)
        
        opcion = input("\nElige una opción (0-3): ").strip()
        
        if opcion == '0':
            # Guardar antes de salir si hay nuevos datos
            if nuevos_datos:
                print("\n¿Guardar los nuevos ejemplos? (s/n): ", end="")
                if input().lower() == 's':
                    df_nuevos = pd.DataFrame(nuevos_datos)
                    df_completo = pd.concat([df_existente, df_nuevos], ignore_index=True)
                    df_completo.to_csv(ruta_datos, index=False)
                    print(f"✅ Guardados {len(nuevos_datos)} nuevos ejemplos")
            print("\n👋 ¡Hasta luego!\n")
            break
        
        elif opcion == '1':
            print("\n" + "="*80)
            print("📄 AGREGAR NUEVO EJEMPLO")
            print("="*80)
            
            # Pedir texto original
            print("\n1️⃣  Pega el TEXTO ORIGINAL (presiona Enter cuando termines):\n")
            print(">>> ", end="", flush=True)
            lineas_texto = []
            try:
                while True:
                    linea = input()
                    if not linea.strip() and lineas_texto:  # Línea vacía después de texto = terminar
                        break
                    if linea.strip():  # Solo agregar líneas no vacías
                        lineas_texto.append(linea)
            except EOFError:
                pass
            
            texto = " ".join(lineas_texto).strip()
            
            if not texto:
                print("⚠️  No ingresaste texto. Intenta de nuevo.")
                continue
            
            # Pedir resumen
            print("\n2️⃣  Ahora escribe el RESUMEN de ese texto (presiona Enter cuando termines):\n")
            print(">>> ", end="", flush=True)
            lineas_resumen = []
            try:
                while True:
                    linea = input()
                    if not linea.strip() and lineas_resumen:  # Línea vacía después de texto = terminar
                        break
                    if linea.strip():  # Solo agregar líneas no vacías
                        lineas_resumen.append(linea)
            except EOFError:
                pass
            
            resumen = " ".join(lineas_resumen).strip()
            
            if not resumen:
                print("⚠️  No ingresaste resumen. Intenta de nuevo.")
                continue
            
            # Mostrar resumen del ejemplo
            print("\n" + "="*80)
            print("✅ EJEMPLO AGREGADO:")
            print("="*80)
            print(f"📄 Texto ({len(texto.split())} palabras):")
            print(texto[:200] + "..." if len(texto) > 200 else texto)
            print(f"\n📝 Resumen ({len(resumen.split())} palabras):")
            print(resumen)
            print("="*80)
            
            # Agregar a la lista
            nuevos_datos.append({
                'texto': texto,
                'resumen': resumen
            })
            
            print(f"\n✅ Ejemplo agregado. Total de nuevos ejemplos: {len(nuevos_datos)}")
        
        elif opcion == '2':
            # Ver ejemplos actuales
            total_ejemplos = len(df_existente) + len(nuevos_datos)
            print(f"\n📊 EJEMPLOS ACTUALES: {total_ejemplos} total")
            print(f"   - Guardados: {len(df_existente)}")
            print(f"   - Nuevos (sin guardar): {len(nuevos_datos)}")
            
            if total_ejemplos > 0:
                print("\n¿Ver detalles? (s/n): ", end="")
                if input().lower() == 's':
                    print("\n" + "="*80)
                    # Mostrar últimos 3 ejemplos
                    todos = pd.concat([df_existente, pd.DataFrame(nuevos_datos)], ignore_index=True)
                    for i, row in todos.tail(3).iterrows():
                        print(f"\nEjemplo {i+1}:")
                        print(f"Texto: {row['texto'][:100]}...")
                        print(f"Resumen: {row['resumen'][:100]}...")
                        print("-"*80)
        
        elif opcion == '3':
            # Entrenar modelo
            total_ejemplos = len(df_existente) + len(nuevos_datos)
            
            if total_ejemplos < 10:
                print(f"\n⚠️  ADVERTENCIA: Solo tienes {total_ejemplos} ejemplos.")
                print("   Se recomienda tener al menos 100 ejemplos para un buen modelo.")
                print("   ¿Continuar de todas formas? (s/n): ", end="")
                if input().lower() != 's':
                    continue
            
            # Guardar datos primero
            if nuevos_datos:
                df_nuevos = pd.DataFrame(nuevos_datos)
                df_completo = pd.concat([df_existente, df_nuevos], ignore_index=True)
            else:
                df_completo = df_existente
            
            # Dividir en train/val (80-20)
            df_train = df_completo.sample(frac=0.8, random_state=42)
            df_val = df_completo.drop(df_train.index)
            
            # Guardar
            config.crear_directorios()
            df_train.to_csv(config.RUTA_DATOS_ENTRENAMIENTO, index=False)
            df_val.to_csv(config.RUTA_DATOS_VALIDACION, index=False)
            
            print(f"\n✅ Datos guardados:")
            print(f"   - Entrenamiento: {len(df_train)} ejemplos")
            print(f"   - Validación: {len(df_val)} ejemplos")
            
            # Ejecutar entrenamiento
            print("\n🚀 Iniciando entrenamiento del modelo...")
            print("   (Esto puede tardar varios minutos)\n")
            
            import subprocess
            resultado = subprocess.run(
                [sys.executable, "src/entrenamiento.py"],
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if resultado.returncode == 0:
                print("\n✅ ¡Entrenamiento completado exitosamente!")
                print("   Ahora puedes usar resumen.py con el modelo entrenado")
            else:
                print("\n❌ Hubo un error en el entrenamiento")
            
            nuevos_datos = []  # Limpiar nuevos datos después de entrenar
        
        else:
            print("\n❌ Opción no válida")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programa interrumpido")
        print("👋 ¡Hasta luego!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
