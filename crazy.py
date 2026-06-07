import re
from pathlib import Path

def escalar_graficos(ruta, factor=1.8):
    p = Path(ruta)
    if not p.exists(): 
        print(f"Archivo no encontrado: {ruta}")
        return
        
    codigo = p.read_text(encoding='utf-8')
    
    # 1. Multiplica automáticamente todas las letras fijadas a mano (fontsize=X, labelsize=X)
    codigo = re.sub(r'(fontsize|labelsize)=(\d+)', 
                    lambda m: f"{m.group(1)}={int(int(m.group(2)) * factor)}", codigo)
    
    # 2. Escala los lienzos físicos. Al reducir el divisor 2.54, el lienzo (figsize) crece proporcionalmente.
    nuevo_divisor = 2.54 / factor
    codigo = codigo.replace("/2.54", f"/{nuevo_divisor:.2f}")
    
    p.write_text(codigo, encoding='utf-8')
    print(f"✅ Escalado masivo (x{factor}) aplicado con éxito en: {p.name}")

# Ejecutar sobre los archivos de tus módulos
for archivo in ["modules/visualize.py", "modules/cluster.py"]:
    escalar_graficos(archivo)