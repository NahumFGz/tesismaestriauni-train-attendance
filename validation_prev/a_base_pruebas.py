import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA_FILE = Path("../data/attendance_docs_enriched.json")

with DATA_FILE.open(encoding="utf-8") as f:
    records = json.load(f)
assert isinstance(records, list)

# Testset original: se usa para EXCLUIR los datos ya seleccionados alli,
# de modo que esta carpeta genere un testset distinto (sin solapamiento).
ORIGINAL_TESTSET_FILE = Path("../validation/testset/preguntas_contexto_esperado.json")

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
MAX_FECHA = 30  # Máximo preguntas por fecha específica
MAX_MES = 10  # Máximo preguntas por mes
MAX_LEGISLATURA = 10  # Máximo preguntas por legislatura

# Semillas para reproducibilidad (distintas a las del testset original)
SEMILLA_FECHA = 1337
SEMILLA_MES = 1337
SEMILLA_LEGISLATURA = 1337


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
# === EXCLUSIONES: datos ya usados en el testset original ===
####################################################
MES_A_NUM = {nombre: i + 1 for i, nombre in enumerate(MESES_ES)}

urls_usadas: set[str] = set()
meses_usados: set[tuple[int, int]] = set()
legislaturas_usadas: set[str] = set()

if ORIGINAL_TESTSET_FILE.exists():
    original = json.loads(ORIGINAL_TESTSET_FILE.read_text(encoding="utf-8"))

    for item in original.get("fecha", []):
        for ctx in item.get("context", []):
            m = re.search(r"URL:\s*(\S+)", ctx)
            if m:
                urls_usadas.add(m.group(1))

    for item in original.get("mes", []):
        m = re.search(r"mes de (\w+) del (\d{4})", item["query"])
        if m and m.group(1) in MES_A_NUM:
            meses_usados.add((int(m.group(2)), MES_A_NUM[m.group(1)]))

    for item in original.get("legislatura", []):
        m = re.match(r"Dame los documentos de asistencia de la (.+)", item["query"])
        if m:
            legislaturas_usadas.add(m.group(1).strip())
    print(
        f"⚠️  Excluyendo datos del testset original: "
        f"{len(urls_usadas)} URLs, {len(meses_usados)} meses, "
        f"{len(legislaturas_usadas)} legislaturas"
    )
else:
    print(f"⚠️  No se encontró el testset original en {ORIGINAL_TESTSET_FILE}; no se excluye nada")

####################################################
# === 1. Por FECHA específica ===
####################################################
records_fecha_disponibles = [r for r in records if r["url"] not in urls_usadas]
random.seed(SEMILLA_FECHA)
records_fecha = random.sample(
    records_fecha_disponibles, k=min(MAX_FECHA, len(records_fecha_disponibles))
)
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

meses_disponibles = [item for item in mes_map.items() if item[0] not in meses_usados]
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

legislaturas = [item for item in leg_map.items() if item[0].strip() not in legislaturas_usadas]
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
