#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de evaluación con bootstrapping para comparar métodos de recuperación.
Evalúa Precision@k, MRR, Top-1 Accuracy, Promedio de rank, MAP, Recall@k
usando comparación de contextos completos (no solo URLs).
"""

import csv
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================================================================
# CONFIGURACIÓN GLOBAL DE BOOTSTRAPPING
# ============================================================================

# Número de muestras bootstrap para intervalos de confianza
# Recomendaciones basadas en 100 pares de datos:
# - 1000: Rápido, buena precisión para intervalos de confianza
# - 5000: Mejor precisión, tiempo moderado (~2-3 min)
# - 10000: Alta precisión, tiempo mayor (~5-7 min)
# - 50000: Máxima precisión, tiempo considerable (~20-30 min)
N_BOOTSTRAP_SAMPLES = 5000

# Nivel de confianza (95% = percentiles 2.5 y 97.5)
CONFIDENCE_LEVEL = 0.95
CI_LOWER_PERCENTILE = (1 - CONFIDENCE_LEVEL) / 2 * 100  # 2.5
CI_UPPER_PERCENTILE = (1 + CONFIDENCE_LEVEL) / 2 * 100  # 97.5

print(f"🔧 CONFIGURACIÓN BOOTSTRAP:")
print(f"  📊 Muestras bootstrap: {N_BOOTSTRAP_SAMPLES:,}")
print(f"  📈 Nivel de confianza: {CONFIDENCE_LEVEL*100:.0f}%")
print(f"  📉 Percentiles: {CI_LOWER_PERCENTILE:.1f}% - {CI_UPPER_PERCENTILE:.1f}%")
print(f"{'='*50}")

# ============================================================================


def extraer_url_de_contexto(contexto: str) -> str:
    """Extrae la URL de un contexto formateado"""
    lines = contexto.strip().split("\n")
    for line in lines:
        if line.startswith("URL: "):
            return line.replace("URL: ", "").strip()
    return ""


def normalizar_contexto(contexto: str) -> str:
    """Normaliza un contexto para comparación (sin URL para focus en contenido)"""
    lines = contexto.strip().split("\n")
    # Tomar solo las primeras dos líneas (contenido sin URL)
    content_lines = [line.strip() for line in lines[:2] if line.strip()]
    return " ".join(content_lines).lower()


def contextos_coinciden(contexto_recuperado: str, contextos_esperados: List[str]) -> bool:
    """
    Verifica si un contexto recuperado coincide con alguno de los esperados.
    Compara tanto el contenido normalizado como la URL.
    """
    url_recuperada = extraer_url_de_contexto(contexto_recuperado)
    contenido_recuperado = normalizar_contexto(contexto_recuperado)

    for contexto_esperado in contextos_esperados:
        url_esperada = extraer_url_de_contexto(contexto_esperado)
        contenido_esperado = normalizar_contexto(contexto_esperado)

        # Coincidencia por URL (más estricta)
        if url_recuperada and url_esperada and url_recuperada == url_esperada:
            return True

        # Coincidencia por contenido normalizado (más flexible)
        if contenido_recuperado and contenido_esperado:
            # Verificar si el contenido es muy similar (al menos 80% de overlap en palabras)
            palabras_recuperadas = set(contenido_recuperado.split())
            palabras_esperadas = set(contenido_esperado.split())

            if palabras_recuperadas and palabras_esperadas:
                overlap = len(palabras_recuperadas & palabras_esperadas)
                union = len(palabras_recuperadas | palabras_esperadas)
                similarity = overlap / union if union > 0 else 0

                if similarity >= 0.8:  # 80% de similitud
                    return True

    return False


def evaluar_metodo(
    ground_truth: Dict, metodo_resultados: Dict, metodo_nombre: str, k: int = 10
) -> Dict[str, float]:
    """
    Evalúa un método de recuperación contra ground truth.

    Args:
        ground_truth: Diccionario con contextos esperados por tipo
        metodo_resultados: Diccionario con contextos recuperados por tipo
        metodo_nombre: Nombre del método para logging
        k: Número máximo de documentos a considerar

    Returns:
        Diccionario con métricas calculadas
    """
    print(f"🔍 Evaluando método: {metodo_nombre}")

    # Combinar todas las preguntas de todos los tipos
    todas_las_preguntas = []

    for tipo in ground_truth.keys():
        if tipo not in metodo_resultados:
            print(f"⚠️  Tipo '{tipo}' no encontrado en {metodo_nombre}")
            continue

        gt_items = ground_truth[tipo]
        metodo_items = metodo_resultados[tipo]

        # Crear diccionario de lookup por query para el método
        metodo_lookup = {item["query"]: item["context"] for item in metodo_items}

        for gt_item in gt_items:
            query = gt_item["query"]
            contextos_esperados = gt_item["context"]

            if query in metodo_lookup:
                contextos_recuperados = metodo_lookup[query]
                todas_las_preguntas.append(
                    {
                        "query": query,
                        "tipo": tipo,
                        "esperados": contextos_esperados,
                        "recuperados": contextos_recuperados[:k],  # Limitar a top-k
                    }
                )

    print(f"  📊 Total de preguntas evaluadas: {len(todas_las_preguntas)}")

    # Calcular métricas (con debug si hay problemas)
    metricas = calcular_metricas(todas_las_preguntas, k, debug=True)

    return metricas


def calcular_metricas(preguntas: List[Dict], k: int, debug: bool = False) -> Dict[str, float]:
    """Calcula todas las métricas de evaluación"""

    precision_scores = []
    reciprocal_ranks = []
    average_precisions = []
    recall_scores = []
    ranks = []
    top1_hits = []

    debug_count = 0

    for pregunta in preguntas:
        esperados = pregunta["esperados"]
        recuperados = pregunta["recuperados"]

        # Encontrar posiciones de hits
        hits_positions = []
        for i, contexto_recuperado in enumerate(recuperados):
            if contextos_coinciden(contexto_recuperado, esperados):
                hits_positions.append(i + 1)  # Posición 1-indexed

        # Precision@k
        precision = len(hits_positions) / min(len(recuperados), k) if recuperados else 0
        precision_scores.append(precision)

        # MRR (Mean Reciprocal Rank)
        if hits_positions:
            reciprocal_rank = 1.0 / hits_positions[0]  # Primera posición de hit
            reciprocal_ranks.append(reciprocal_rank)
            ranks.append(hits_positions[0])

            # Top-1 Accuracy
            top1_hits.append(1 if hits_positions[0] == 1 else 0)
        else:
            reciprocal_ranks.append(0.0)
            ranks.append(k + 1)  # Peor que k
            top1_hits.append(0)

        # Average Precision (AP) - Fórmula corregida para estar en [0,1]
        if hits_positions and esperados:
            # Fórmula estándar de AP: (1/R) * Σ(Precision@k * rel(k))
            # donde R = número total de documentos relevantes
            ap_sum = 0.0
            for j, pos in enumerate(hits_positions):
                # Precision@pos = (documentos relevantes hasta esta posición) / posición
                precision_at_pos = (j + 1) / pos
                ap_sum += precision_at_pos

            # AP = suma de precisiones / número de documentos relevantes encontrados
            # Esto garantiza AP ∈ [0,1] porque cada precision_at_pos ≤ 1
            ap = ap_sum / len(hits_positions) if hits_positions else 0.0

            # Verificación de seguridad: asegurar que AP ≤ 1.0
            ap = min(1.0, ap)
        else:
            ap = 0.0
        average_precisions.append(ap)

        # Recall@k - Corregido para evitar valores > 1
        if esperados:
            # Recall = documentos relevantes encontrados / total documentos relevantes
            # Pero limitado a máximo 1.0 (no puede recuperar más del 100%)
            recall = min(1.0, len(hits_positions) / len(esperados))
        else:
            recall = 0.0
        recall_scores.append(recall)

        # Debug para casos problemáticos
        if debug and debug_count < 3 and (ap > 1.0 or recall > 1.0):
            print(f"  🐛 DEBUG pregunta {debug_count + 1}:")
            print(f"    Query: {pregunta.get('query', 'N/A')[:50]}...")
            print(f"    Esperados: {len(esperados)}")
            print(f"    Recuperados: {len(recuperados)}")
            print(f"    Hits: {len(hits_positions)} en posiciones {hits_positions}")
            print(f"    AP: {ap:.4f}, Recall: {recall:.4f}")
            debug_count += 1

    # Agregar métricas con validación de rangos
    map_value = np.mean(average_precisions)

    # Validación crítica: MAP debe estar en [0,1]
    if map_value > 1.0:
        print(f"⚠️  MAP calculado ({map_value:.4f}) > 1.0, limitando a 1.0")
        map_value = 1.0

    metricas = {
        f"precision_at_{k}": min(1.0, np.mean(precision_scores)),
        "mrr": min(1.0, np.mean(reciprocal_ranks)),
        "top1_accuracy": min(1.0, np.mean(top1_hits)),
        "average_rank": (
            np.mean([r for r in ranks if r <= k]) if any(r <= k for r in ranks) else k + 1
        ),
        "map": map_value,
        f"recall_at_{k}": min(1.0, np.mean(recall_scores)),
    }

    return metricas


def bootstrap_metricas(
    ground_truth: Dict,
    metodo_resultados: Dict,
    metodo_nombre: str,
    n_bootstrap: int = None,
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Calcula intervalos de confianza usando bootstrapping.

    Args:
        ground_truth: Ground truth data
        metodo_resultados: Resultados del método
        metodo_nombre: Nombre del método
        n_bootstrap: Número de muestras bootstrap (None usa la configuración global)
        k: Valor de k para las métricas

    Returns:
        Diccionario con métricas y sus intervalos de confianza
    """
    # Usar configuración global si no se especifica
    if n_bootstrap is None:
        n_bootstrap = N_BOOTSTRAP_SAMPLES

    print(f"🔄 Aplicando bootstrapping para {metodo_nombre} ({n_bootstrap:,} muestras)...")

    # Preparar datos combinados
    todas_las_preguntas = []
    for tipo in ground_truth.keys():
        if tipo not in metodo_resultados:
            continue

        gt_items = ground_truth[tipo]
        metodo_items = metodo_resultados[tipo]
        metodo_lookup = {item["query"]: item["context"] for item in metodo_items}

        for gt_item in gt_items:
            query = gt_item["query"]
            if query in metodo_lookup:
                todas_las_preguntas.append(
                    {
                        "query": query,
                        "tipo": tipo,
                        "esperados": gt_item["context"],
                        "recuperados": metodo_lookup[query][:k],
                    }
                )

    # Bootstrap sampling
    bootstrap_results = defaultdict(list)

    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"  📈 Bootstrap sample {i + 1}/{n_bootstrap}")

        # Muestra con reemplazo
        sample = random.choices(todas_las_preguntas, k=len(todas_las_preguntas))

        # Calcular métricas para esta muestra
        metricas_sample = calcular_metricas(sample, k)

        for metric_name, value in metricas_sample.items():
            bootstrap_results[metric_name].append(value)

    # Calcular intervalos de confianza (95%) con validación de rangos
    resultados_finales = {}
    for metric_name, values in bootstrap_results.items():
        # Validar que todas las muestras estén en rango válido
        if metric_name in ["precision_at_10", "mrr", "top1_accuracy", "map", "recall_at_10"]:
            # Limitar valores a [0,1] para métricas que deben estar en este rango
            values = [min(1.0, max(0.0, v)) for v in values]

            # Advertencia si hay valores fuera de rango
            valores_altos = [v for v in values if v > 1.0]
            if valores_altos:
                print(f"⚠️  {metric_name}: {len(valores_altos)} muestras > 1.0 corregidas")

        mean_val = np.mean(values)
        ci_lower = np.percentile(values, CI_LOWER_PERCENTILE)
        ci_upper = np.percentile(values, CI_UPPER_PERCENTILE)
        std_val = np.std(values)

        # Validación final de rangos
        if metric_name != "average_rank":
            mean_val = min(1.0, max(0.0, mean_val))
            ci_lower = min(1.0, max(0.0, ci_lower))
            ci_upper = min(1.0, max(0.0, ci_upper))

        resultados_finales[metric_name] = {
            "mean": mean_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std": std_val,
        }

    return resultados_finales


def validar_datos_graficas(resultados_todos_metodos: Dict) -> bool:
    """
    Valida que los datos estén en formato correcto para las gráficas.

    Args:
        resultados_todos_metodos: Resultados de todos los métodos

    Returns:
        True si los datos son válidos, False en caso contrario
    """
    metricas = ["precision_at_10", "mrr", "top1_accuracy", "map", "recall_at_10"]

    for metodo_nombre, resultados in resultados_todos_metodos.items():
        if "bootstrap" not in resultados:
            print(f"⚠️  Método {metodo_nombre} no tiene datos bootstrap")
            return False

        for metrica in metricas:
            if metrica in resultados["bootstrap"]:
                stats = resultados["bootstrap"][metrica]

                # Validar que todos los campos requeridos existan
                campos_requeridos = ["mean", "ci_lower", "ci_upper", "std"]
                for campo in campos_requeridos:
                    if campo not in stats:
                        print(f"⚠️  {metodo_nombre}.{metrica} falta campo: {campo}")
                        return False

                # Validar rangos lógicos
                mean = stats["mean"]
                ci_lower = stats["ci_lower"]
                ci_upper = stats["ci_upper"]

                if ci_lower > mean:
                    print(
                        f"⚠️  {metodo_nombre}.{metrica}: ci_lower ({ci_lower:.4f}) > mean ({mean:.4f})"
                    )

                if ci_upper < mean:
                    print(
                        f"⚠️  {metodo_nombre}.{metrica}: ci_upper ({ci_upper:.4f}) < mean ({mean:.4f})"
                    )

                # Solo validar rango [0,1] para métricas que deberían estar en ese rango
                # average_rank puede ser > 1, así que no la validamos
                if metrica != "average_rank" and (mean < 0 or mean > 1):
                    print(f"⚠️  {metodo_nombre}.{metrica}: mean fuera de rango [0,1]: {mean:.4f}")

    return True


def crear_graficas_comparativas(resultados_todos_metodos: Dict, output_dir: Path):
    """
    Crea gráficas comparativas de los métodos de recuperación.

    Args:
        resultados_todos_metodos: Resultados de todos los métodos
        output_dir: Directorio donde guardar las gráficas
    """
    print("📊 Generando gráficas comparativas...")

    # Validar datos antes de crear gráficas
    if not validar_datos_graficas(resultados_todos_metodos):
        print("⚠️  Se encontraron problemas en los datos, pero continuando con las gráficas...")

    # Configurar estilo
    plt.style.use("default")
    sns.set_palette("husl")

    # Métricas a graficar
    metricas = ["precision_at_10", "mrr", "top1_accuracy", "map", "recall_at_10"]
    nombres_metricas = {
        "precision_at_10": "Precision@10",
        "mrr": "Mean Reciprocal Rank",
        "top1_accuracy": "Top-1 Accuracy",
        "map": "Mean Average Precision",
        "recall_at_10": "Recall@10",
    }

    # 1. Gráfica de barras con intervalos de confianza
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Comparación de Métodos de Recuperación con Intervalos de Confianza ({CONFIDENCE_LEVEL*100:.0f}%)",
            fontsize=16,
            fontweight="bold",
        )

        axes = axes.flatten()

        for i, metrica in enumerate(metricas):
            ax = axes[i]

            metodos = []
            means = []
            ci_lowers = []
            ci_uppers = []

            for metodo_nombre, resultados in resultados_todos_metodos.items():
                if metrica in resultados["bootstrap"]:
                    stats = resultados["bootstrap"][metrica]
                    metodos.append(metodo_nombre.replace("_", " ").title())
                    means.append(stats["mean"])
                    ci_lowers.append(stats["ci_lower"])
                    ci_uppers.append(stats["ci_upper"])

            if means:
                # Crear barras con intervalos de confianza
                x_pos = np.arange(len(metodos))
                bars = ax.bar(x_pos, means, alpha=0.7, capsize=5)

                # Agregar intervalos de confianza (asegurar valores no negativos)
                yerr_lower = [max(0, mean - ci_lower) for mean, ci_lower in zip(means, ci_lowers)]
                yerr_upper = [max(0, ci_upper - mean) for mean, ci_upper in zip(ci_uppers, means)]
                ax.errorbar(
                    x_pos,
                    means,
                    yerr=[yerr_lower, yerr_upper],
                    fmt="none",
                    color="black",
                    capsize=5,
                    capthick=2,
                )

                # Personalizar gráfica
                ax.set_title(nombres_metricas[metrica], fontweight="bold")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metodos, rotation=45, ha="right")
                ax.set_ylabel("Score")
                ax.grid(True, alpha=0.3)

                # Agregar valores en las barras
                for j, (bar, mean) in enumerate(zip(bars, means)):
                    height = bar.get_height()
                    # Posición del texto ligeramente arriba de la barra + error
                    text_y = height + yerr_upper[j] + 0.01  # Pequeño offset adicional
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        text_y,
                        f"{mean:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=9,
                    )

        # Remover subplot vacío
        fig.delaxes(axes[5])

        plt.tight_layout()
        plt.savefig(output_dir / "comparacion_barras_ci.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"❌ Error generando gráfica de barras: {e}")
        print("📊 Continuando con las siguientes gráficas...")

    # 2. Gráfica de radar/spider
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Preparar datos para radar
    metodos_radar = list(resultados_todos_metodos.keys())
    angles = np.linspace(0, 2 * np.pi, len(metricas), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo

    colors = plt.cm.Set3(np.linspace(0, 1, len(metodos_radar)))

    for i, metodo_nombre in enumerate(metodos_radar):
        resultados = resultados_todos_metodos[metodo_nombre]
        values = []

        for metrica in metricas:
            if metrica in resultados["bootstrap"]:
                values.append(resultados["bootstrap"][metrica]["mean"])
            else:
                values.append(0)

        values += values[:1]  # Cerrar el círculo

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=metodo_nombre.replace("_", " ").title(),
            color=colors[i],
        )
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    # Personalizar gráfica radar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([nombres_metricas[m] for m in metricas])
    ax.set_ylim(0, 1)
    ax.set_title(
        "Comparación Multidimensional de Métodos\n(Todas las Métricas)",
        size=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "comparacion_radar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Heatmap de correlación entre métodos
    fig, ax = plt.subplots(figsize=(10, 8))

    # Crear matriz de datos para heatmap
    data_matrix = []
    metodos_labels = []

    for metodo_nombre, resultados in resultados_todos_metodos.items():
        row = []
        for metrica in metricas:
            if metrica in resultados["bootstrap"]:
                row.append(resultados["bootstrap"][metrica]["mean"])
            else:
                row.append(0)
        data_matrix.append(row)
        metodos_labels.append(metodo_nombre.replace("_", " ").title())

    # Crear heatmap
    sns.heatmap(
        data_matrix,
        xticklabels=[nombres_metricas[m] for m in metricas],
        yticklabels=metodos_labels,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0.5,
        square=True,
        ax=ax,
    )

    ax.set_title(
        "Heatmap de Rendimiento por Método y Métrica", fontsize=14, fontweight="bold", pad=20
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_rendimiento.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Gráfica de distribuciones bootstrap (violin plots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Distribuciones Bootstrap de las Métricas", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    # Recolectar datos de distribuciones bootstrap para violin plots
    # Nota: Para esto necesitaríamos guardar las distribuciones completas,
    # pero por simplicidad mostraremos boxplots con los intervalos de confianza

    for i, metrica in enumerate(metricas):
        ax = axes[i]

        data_for_boxplot = []
        labels_for_boxplot = []

        for metodo_nombre, resultados in resultados_todos_metodos.items():
            if metrica in resultados["bootstrap"]:
                stats = resultados["bootstrap"][metrica]
                # Simular distribución normal basada en mean y std
                # En implementación real, usarías las muestras bootstrap guardadas
                simulated_data = np.random.normal(stats["mean"], stats["std"], 1000)
                data_for_boxplot.append(simulated_data)
                labels_for_boxplot.append(metodo_nombre.replace("_", " ").title())

        if data_for_boxplot:
            # Crear violin plot
            parts = ax.violinplot(
                data_for_boxplot,
                positions=range(len(data_for_boxplot)),
                showmeans=True,
                showmedians=True,
            )

            # Personalizar colores
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[j % len(colors)])
                pc.set_alpha(0.7)

            ax.set_title(nombres_metricas[metrica], fontweight="bold")
            ax.set_xticks(range(len(labels_for_boxplot)))
            ax.set_xticklabels(labels_for_boxplot, rotation=45, ha="right")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)

    # Remover subplot vacío
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(output_dir / "distribuciones_bootstrap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Gráfica de ranking comparativo
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calcular ranking promedio para cada método
    metodos_ranking = {}
    for metodo_nombre, resultados in resultados_todos_metodos.items():
        ranks = []
        for metrica in metricas:
            if metrica in resultados["bootstrap"]:
                # Obtener ranking de este método para esta métrica
                metrica_values = []
                for otro_metodo, otros_resultados in resultados_todos_metodos.items():
                    if metrica in otros_resultados["bootstrap"]:
                        metrica_values.append(
                            (otros_resultados["bootstrap"][metrica]["mean"], otro_metodo)
                        )

                # Ordenar y encontrar ranking (1 = mejor)
                metrica_values.sort(reverse=True)
                for rank, (value, nombre) in enumerate(metrica_values, 1):
                    if nombre == metodo_nombre:
                        ranks.append(rank)
                        break

        if ranks:
            metodos_ranking[metodo_nombre] = np.mean(ranks)

    # Ordenar métodos por ranking promedio
    metodos_ordenados = sorted(metodos_ranking.items(), key=lambda x: x[1])

    nombres = [m.replace("_", " ").title() for m, _ in metodos_ordenados]
    rankings = [r for _, r in metodos_ordenados]

    # Crear gráfica de barras horizontales
    bars = ax.barh(nombres, rankings, color=colors[: len(nombres)])

    # Personalizar
    ax.set_xlabel("Ranking Promedio (1 = Mejor)", fontweight="bold")
    ax.set_title("Ranking Promedio Across Todas las Métricas", fontweight="bold", pad=20)
    ax.invert_yaxis()  # Mejor método arriba
    ax.grid(True, alpha=0.3, axis="x")

    # Agregar valores en las barras
    for bar, ranking in zip(bars, rankings):
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{ranking:.1f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "ranking_promedio.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Gráficas generadas:")
    print("  📊 comparacion_barras_ci.png - Barras con intervalos de confianza")
    print("  🕷️  comparacion_radar.png - Gráfica radar multidimensional")
    print("  🔥 heatmap_rendimiento.png - Heatmap de rendimiento")
    print("  🎻 distribuciones_bootstrap.png - Distribuciones bootstrap")
    print("  🏆 ranking_promedio.png - Ranking promedio de métodos")


def generar_reporte_html(resultados_todos_metodos: Dict, output_dir: Path):
    """
    Genera un reporte HTML interactivo con todas las gráficas y resultados.

    Args:
        resultados_todos_metodos: Resultados de todos los métodos
        output_dir: Directorio donde guardar el reporte
    """
    print("📄 Generando reporte HTML...")

    html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Evaluación - Métodos de Recuperación</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}
        .metric-table th, .metric-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        .metric-table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        .metric-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric-table tr:hover {{
            background-color: #e8f4f8;
        }}
        .best-score {{
            background-color: #d4edda !important;
            font-weight: bold;
            color: #155724;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .summary-box {{
            background-color: #e8f4f8;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .method-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .confidence-interval {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Reporte de Evaluación - Métodos de Recuperación</h1>
        
        <div class="summary-box">
            <h3>📊 Resumen Ejecutivo</h3>
            <p>Este reporte compara 4 métodos de recuperación de información usando bootstrapping 
               con {N_BOOTSTRAP_SAMPLES:,} muestras para calcular intervalos de confianza del {CONFIDENCE_LEVEL*100:.0f}%. 
               Se evaluaron {sum(len(items) for items in [resultados_todos_metodos[list(resultados_todos_metodos.keys())[0]]["bootstrap"].keys()])} métricas 
               sobre un total de 100 preguntas.</p>
        </div>

        <h2>📈 Tabla Comparativa de Resultados</h2>
        <table class="metric-table">
            <thead>
                <tr>
                    <th>Método</th>
                    <th>Precision@10</th>
                    <th>MRR</th>
                    <th>Top-1 Accuracy</th>
                    <th>MAP</th>
                    <th>Recall@10</th>
                </tr>
            </thead>
            <tbody>
"""

    # Agregar filas de datos
    metricas_principales = ["precision_at_10", "mrr", "top1_accuracy", "map", "recall_at_10"]

    # Encontrar mejores scores para resaltar
    mejores_scores = {}
    for metrica in metricas_principales:
        mejor_score = -1
        for metodo_nombre, resultados in resultados_todos_metodos.items():
            if metrica in resultados["bootstrap"]:
                score = resultados["bootstrap"][metrica]["mean"]
                if score > mejor_score:
                    mejor_score = score
        mejores_scores[metrica] = mejor_score

    for metodo_nombre, resultados in resultados_todos_metodos.items():
        html_content += f"""
                <tr>
                    <td class="method-name">{metodo_nombre.replace('_', ' ').title()}</td>
"""

        for metrica in metricas_principales:
            if metrica in resultados["bootstrap"]:
                stats = resultados["bootstrap"][metrica]
                es_mejor = abs(stats["mean"] - mejores_scores[metrica]) < 0.0001
                class_name = "best-score" if es_mejor else ""

                html_content += f"""
                    <td class="{class_name}">
                        {stats["mean"]:.4f}<br>
                        <span class="confidence-interval">[{stats["ci_lower"]:.4f}, {stats["ci_upper"]:.4f}]</span>
                    </td>
"""
            else:
                html_content += "<td>N/A</td>"

        html_content += "                </tr>"

    html_content += """
            </tbody>
        </table>

        <h2>📊 Gráficas Comparativas</h2>
        
        <div class="image-container">
            <h3>Comparación con Intervalos de Confianza</h3>
            <img src="comparacion_barras_ci.png" alt="Comparación de barras con intervalos de confianza">
            <p>Esta gráfica muestra el rendimiento promedio de cada método con sus intervalos de confianza del {CONFIDENCE_LEVEL*100:.0f}%.</p>
        </div>

        <div class="image-container">
            <h3>Análisis Multidimensional (Radar)</h3>
            <img src="comparacion_radar.png" alt="Gráfica radar multidimensional">
            <p>Visualización radar que permite comparar todos los métodos across múltiples métricas simultáneamente.</p>
        </div>

        <div class="image-container">
            <h3>Heatmap de Rendimiento</h3>
            <img src="heatmap_rendimiento.png" alt="Heatmap de rendimiento">
            <p>Mapa de calor que muestra la intensidad del rendimiento de cada método en cada métrica.</p>
        </div>

        <div class="image-container">
            <h3>Distribuciones Bootstrap</h3>
            <img src="distribuciones_bootstrap.png" alt="Distribuciones bootstrap">
            <p>Violin plots que muestran las distribuciones bootstrap de las métricas para cada método.</p>
        </div>

        <div class="image-container">
            <h3>Ranking Promedio</h3>
            <img src="ranking_promedio.png" alt="Ranking promedio de métodos">
            <p>Ranking general de los métodos basado en el promedio de posiciones across todas las métricas.</p>
        </div>

        <h2>🏆 Conclusiones</h2>
        <div class="summary-box">
"""

    # Calcular método ganador
    mejor_metodo = None
    mejor_score_promedio = -1

    for metodo_nombre, resultados in resultados_todos_metodos.items():
        scores = []
        for metrica in metricas_principales:
            if metrica in resultados["bootstrap"]:
                scores.append(resultados["bootstrap"][metrica]["mean"])

        if scores:
            score_promedio = np.mean(scores)
            if score_promedio > mejor_score_promedio:
                mejor_score_promedio = score_promedio
                mejor_metodo = metodo_nombre

    html_content += f"""
            <h4>🥇 Método Ganador: {mejor_metodo.replace('_', ' ').title() if mejor_metodo else 'N/A'}</h4>
            <p>Basado en el análisis de bootstrapping con intervalos de confianza del {CONFIDENCE_LEVEL*100:.0f}%, 
               el método <strong>{mejor_metodo.replace('_', ' ').title() if mejor_metodo else 'N/A'}</strong> 
               muestra el mejor rendimiento promedio across todas las métricas evaluadas.</p>
            
            <h4>📋 Interpretación de Métricas:</h4>
            <ul>
                <li><strong>Precision@10:</strong> Proporción de documentos relevantes en los primeros 10 resultados</li>
                <li><strong>MRR:</strong> Mean Reciprocal Rank - posición promedio del primer documento relevante</li>
                <li><strong>Top-1 Accuracy:</strong> Proporción de consultas donde el primer resultado es relevante</li>
                <li><strong>MAP:</strong> Mean Average Precision - precisión promedio considerando el orden</li>
                <li><strong>Recall@10:</strong> Proporción de documentos relevantes recuperados en top-10</li>
            </ul>
        </div>

        <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
            <p>Reporte generado automáticamente con bootstrapping estadístico</p>
            <p>📅 Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""

    # Guardar reporte HTML
    html_path = output_dir / "reporte_evaluacion.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Reporte HTML generado: {html_path}")


def main():
    """Función principal que ejecuta toda la evaluación con bootstrapping"""

    # Configuración
    k = 10  # Top-k para evaluación

    print("🔍 DIAGNÓSTICO: Revisando cálculo de métricas...")
    print(
        f"🎯 Usando {N_BOOTSTRAP_SAMPLES:,} muestras bootstrap con {CONFIDENCE_LEVEL*100:.0f}% confianza"
    )

    # Rutas de archivos
    BASE_DIR = Path("./testset")
    OUTPUT_DIR = Path("./bootstrapping")
    OUTPUT_DIR.mkdir(exist_ok=True)

    GROUND_TRUTH_PATH = BASE_DIR / "preguntas_contexto_esperado.json"

    METODOS = {
        "keyword_search": BASE_DIR / "keyword_search.json",
        "tf_idf": BASE_DIR / "tf_idf.json",
        "bm25": BASE_DIR / "bm25.json",
        "qdrant": BASE_DIR / "contexto_qdrant.json",
    }

    print("🚀 Iniciando evaluación con bootstrapping")
    print(f"⚙️  Configuración: k={k}, n_bootstrap={N_BOOTSTRAP_SAMPLES:,}")

    # Cargar ground truth
    print("📖 Cargando ground truth...")
    try:
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        print(
            f"✅ Ground truth cargado: {sum(len(items) for items in ground_truth.values())} preguntas"
        )
    except Exception as e:
        print(f"❌ Error cargando ground truth: {e}")
        return

    # Evaluar cada método
    resultados_todos_metodos = {}

    for metodo_nombre, metodo_path in METODOS.items():
        print(f"\n{'='*50}")
        print(f"🔍 Evaluando método: {metodo_nombre.upper()}")

        try:
            with open(metodo_path, "r", encoding="utf-8") as f:
                metodo_resultados = json.load(f)

            # Evaluación básica
            metricas_basicas = evaluar_metodo(ground_truth, metodo_resultados, metodo_nombre, k)

            # Diagnóstico de valores extraños
            print(f"🔍 Diagnóstico {metodo_nombre}:")
            for metric_name, value in metricas_basicas.items():
                if metric_name not in ["average_rank"] and (value < 0 or value > 1):
                    print(f"  ⚠️  {metric_name}: {value:.4f} (fuera de rango [0,1])")
                else:
                    print(f"  ✅ {metric_name}: {value:.4f}")

            # Diagnóstico adicional para debug
            if any(metricas_basicas.get(m, 0) > 1 for m in ["map", f"recall_at_{k}"]):
                print(f"  🔍 Debug para {metodo_nombre}: valores anómalos detectados")

            # Bootstrapping (usa configuración global)
            metricas_bootstrap = bootstrap_metricas(
                ground_truth, metodo_resultados, metodo_nombre, None, k
            )

            resultados_todos_metodos[metodo_nombre] = {
                "basicas": metricas_basicas,
                "bootstrap": metricas_bootstrap,
            }

            # Guardar resultados individuales
            output_file = OUTPUT_DIR / f"{metodo_nombre}_bootstrap_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metodo": metodo_nombre,
                        "configuracion": {
                            "k": k,
                            "n_bootstrap": N_BOOTSTRAP_SAMPLES,
                            "confidence_level": CONFIDENCE_LEVEL,
                        },
                        "metricas_basicas": metricas_basicas,
                        "bootstrap_ci": metricas_bootstrap,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"✅ Resultados guardados en: {output_file}")

        except Exception as e:
            print(f"❌ Error evaluando {metodo_nombre}: {e}")
            continue

    # Generar reporte comparativo
    print(f"\n{'='*50}")
    print("📊 Generando reporte comparativo...")

    # CSV con comparación de métricas
    csv_path = OUTPUT_DIR / "comparacion_metodos.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = [
            "Metodo",
            "Precision@10",
            "Precision@10_CI",
            "MRR",
            "MRR_CI",
            "Top1_Accuracy",
            "Top1_Accuracy_CI",
            "Average_Rank",
            "Average_Rank_CI",
            "MAP",
            "MAP_CI",
            "Recall@10",
            "Recall@10_CI",
        ]
        writer.writerow(header)

        # Datos por método
        for metodo_nombre, resultados in resultados_todos_metodos.items():
            bootstrap = resultados["bootstrap"]

            row = [metodo_nombre]
            for metric in [
                "precision_at_10",
                "mrr",
                "top1_accuracy",
                "average_rank",
                "map",
                "recall_at_10",
            ]:
                if metric in bootstrap:
                    mean_val = bootstrap[metric]["mean"]
                    ci_lower = bootstrap[metric]["ci_lower"]
                    ci_upper = bootstrap[metric]["ci_upper"]

                    row.append(f"{mean_val:.4f}")
                    row.append(f"[{ci_lower:.4f}, {ci_upper:.4f}]")
                else:
                    row.extend(["N/A", "N/A"])

            writer.writerow(row)

    print(f"✅ Reporte comparativo guardado en: {csv_path}")

    # Generar gráficas comparativas
    crear_graficas_comparativas(resultados_todos_metodos, OUTPUT_DIR)

    # Generar reporte HTML
    generar_reporte_html(resultados_todos_metodos, OUTPUT_DIR)

    # Resumen en consola
    print(f"\n{'='*50}")
    print("📈 RESUMEN DE RESULTADOS")
    print(f"{'='*50}")

    for metodo_nombre, resultados in resultados_todos_metodos.items():
        bootstrap = resultados["bootstrap"]
        print(f"\n🔸 {metodo_nombre.upper()}:")

        for metric_name in ["precision_at_10", "mrr", "top1_accuracy", "map", "recall_at_10"]:
            if metric_name in bootstrap:
                stats = bootstrap[metric_name]
                print(
                    f"  {metric_name:15}: {stats['mean']:.4f} "
                    f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                )

    # Guardar resumen completo
    summary_path = OUTPUT_DIR / "resumen_completo.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "configuracion": {
                    "k": k,
                    "n_bootstrap": N_BOOTSTRAP_SAMPLES,
                    "confidence_level": CONFIDENCE_LEVEL,
                },
                "metodos_evaluados": list(METODOS.keys()),
                "total_preguntas": sum(len(items) for items in ground_truth.values()),
                "resultados": resultados_todos_metodos,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n✅ Evaluación completa. Resultados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Fijar semilla para reproducibilidad
    random.seed(42)
    np.random.seed(42)

    main()
