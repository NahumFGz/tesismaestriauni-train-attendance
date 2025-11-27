import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA_FILE = Path("../data/attendance_docs_enriched.json")

with DATA_FILE.open(encoding="utf-8") as f:
    records = json.load(f)
assert isinstance(records, list)

MESES_ES = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]

# Configuración global - Variables por categoría
MAX_FECHA = 80  # Máximo preguntas por fecha específica
MAX_MES = 10  # Máximo preguntas por mes
MAX_LEGISLATURA = 10  # Máximo preguntas por legislatura

# Semillas para reproducibilidad
SEMILLA_FECHA = 42
SEMILLA_MES = 42
SEMILLA_LEGISLATURA = 42


def human_date_es(iso_date: str) -> str:
    dt = datetime.fromisoformat(iso_date)
    return f"{dt.day} de {MESES_ES[dt.month - 1]} de {dt.year}"


def format_document_context(record: dict) -> str:
    """Formatea un documento con el mismo formato usado en la indexación de Qdrant"""
    return (
        f"Asistencia del {record['fecha_larga']} – {record['legislatura']}.\n"
        f"Congreso {record['periodo_congreso']} | "
        f"Periodo anual {record['periodo_anual']}.\n"
        f"URL: {record['url']}"
    )


# Diccionario principal para todas las categorías
all_queries = {}

####################################################
# === 1. Por FECHA específica ===
####################################################
random.seed(SEMILLA_FECHA)
records_fecha = random.sample(records, k=min(MAX_FECHA, len(records)))
out_fecha = []
for rec in records_fecha:
    pregunta = f"¿Cuál fue la asistencia del Congreso el {human_date_es(rec['fecha_utc5'])}?"
    context = [format_document_context(rec)]
    out_fecha.append({"query": pregunta, "context": context})
all_queries["fecha"] = out_fecha

####################################################
# === 2. Por MES ===
####################################################
mes_map = defaultdict(list)
for r in records:
    dt = datetime.fromisoformat(r["fecha_utc5"])
    clave = (dt.year, dt.month)
    mes_map[clave].append(r)

meses_disponibles = list(mes_map.items())
random.seed(SEMILLA_MES)
random.shuffle(meses_disponibles)
out_mes = []
for (año, mes), records_mes in meses_disponibles[:MAX_MES]:
    pregunta = f"Dame la asistencia del mes de {MESES_ES[mes - 1]} del {año}"
    context = [format_document_context(rec) for rec in records_mes]
    out_mes.append({"query": pregunta, "context": context})
all_queries["mes"] = out_mes

####################################################
# === 3. Por LEGISLATURA ===
####################################################
leg_map = defaultdict(list)
for r in records:
    leg_map[r["legislatura"]].append(r)

legislaturas = list(leg_map.items())
random.seed(SEMILLA_LEGISLATURA)
random.shuffle(legislaturas)
out_leg = []
for legislatura, records_leg in legislaturas[:MAX_LEGISLATURA]:
    pregunta = f"Dame los documentos de asistencia de la {legislatura}"
    context = [format_document_context(rec) for rec in records_leg]
    out_leg.append({"query": pregunta, "context": context})
all_queries["legislatura"] = out_leg


####################################################
# === GUARDAR ARCHIVO ÚNICO ===
####################################################
output_file = Path("./testset/preguntas_contexto_esperado.json")
output_file.write_text(json.dumps(all_queries, ensure_ascii=False, indent=2), encoding="utf-8")

# Estadísticas
print("✅ Generadas bases de prueba para 3 categorías:")
for categoria, queries in all_queries.items():
    print(f"  - {categoria}: {len(queries)} preguntas")

print(f"\n📁 Archivo generado: {output_file}")
print(f"📊 Total de preguntas: {sum(len(queries) for queries in all_queries.values())}")

# Mostrar ejemplo de estructura
print("\n📋 Estructura del JSON generado:")
for categoria in all_queries.keys():
    print(f"  - {categoria}")
    if all_queries[categoria]:
        ejemplo = all_queries[categoria][0]
        print(f"    Ejemplo: {ejemplo['query'][:60]}...")

print(f"\n🔍 Estadísticas de agrupación:")
print(f"  - Legislaturas distintas: {len(leg_map)}")
print(f"  - Meses disponibles: {len(mes_map)}")
print(f"  - Total de documentos: {len(records)}")
print(f"\n⚙️  Configuración:")
print(f"  - MAX_FECHA = {MAX_FECHA} (semilla: {SEMILLA_FECHA})")
print(f"  - MAX_MES = {MAX_MES} (semilla: {SEMILLA_MES})")
print(f"  - MAX_LEGISLATURA = {MAX_LEGISLATURA} (semilla: {SEMILLA_LEGISLATURA})")
