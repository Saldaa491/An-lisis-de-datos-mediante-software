"""
============================================================
TAREA 4 - SIMULACIÓN DE SISTEMAS CUANTITATIVOS
Predicción de Rotación de Empleados (Employee Attrition)
Dataset: IBM HR Analytics Employee Attrition
============================================================
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------- CONFIGURACIÓN GLOBAL ----------
RANDOM_STATE = 42
TARGET = "Attrition"

# Estilo de las gráficas
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100


# ---------- FUNCIONES AUXILIARES ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Analiza y predice la rotación de empleados."
    )
    ruta_predeterminada = Path(__file__).resolve().parent / "HR-Employee-Attrition.csv"
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(ruta_predeterminada),
        help="Ruta del archivo CSV (por defecto, junto a este script).",
    )
    return parser.parse_args()


def cargar_datos(ruta_csv):
    ruta = Path(ruta_csv).expanduser().resolve()
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo CSV:\n  {ruta}\n"
            "Coloca HR-Employee-Attrition.csv junto a ejercicio.py o indica su ruta."
        )

    datos = pd.read_csv(ruta)
    if TARGET not in datos.columns:
        raise ValueError(f"El CSV debe incluir una columna llamada '{TARGET}'.")
    if datos.empty:
        raise ValueError("El archivo CSV no contiene filas de datos.")

    attrition = datos[TARGET].astype("string").str.strip().str.lower()
    valores_validos = {"yes", "no"}
    valores_encontrados = set(attrition.dropna().unique())
    if not valores_encontrados.issubset(valores_validos) or attrition.isna().any():
        raise ValueError(
            f"La columna '{TARGET}' debe contener solamente Yes y No. "
            f"Valores encontrados: {sorted(valores_encontrados)}"
        )
    datos[TARGET] = attrition.map({"yes": "Yes", "no": "No"})
    return datos


# ---------- ANÁLISIS EXPLORATORIO ----------
def mostrar_analisis_exploratorio(datos):
    print("\n" + "=" * 60)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 60)

    # Distribución de Attrition
    print(f"\nDistribución de Attrition:\n{datos[TARGET].value_counts()}")
    porcentaje = datos[TARGET].eq("Yes").mean() * 100
    print(f"Porcentaje de rotación: {porcentaje:.2f}%")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=datos, x=TARGET, hue=TARGET, palette="Set2", legend=False)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.title("Distribución de Rotación de Empleados", fontsize=14, fontweight='bold')
    plt.xlabel("Rotación")
    plt.ylabel("Cantidad de Empleados")
    plt.tight_layout()
    plt.show()

    # Boxplots de variables numéricas
    columnas_numericas = [
        col for col in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"]
        if col in datos.columns
    ]
    if columnas_numericas:
        filas = int(np.ceil(len(columnas_numericas) / 2))
        fig, ejes = plt.subplots(filas, 2, figsize=(12, 4 * filas))
        for eje, col in zip(np.asarray(ejes).reshape(-1), columnas_numericas):
            sns.boxplot(data=datos, x=TARGET, y=col, hue=TARGET, palette="Set1", legend=False, ax=eje)
            eje.set_title(f"{col} según rotación", fontweight='bold')
        for eje in np.asarray(ejes).reshape(-1)[len(columnas_numericas):]:
            eje.remove()
        fig.suptitle("Variables Numéricas vs Rotación", fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.show()

    # Countplots de variables categóricas
    columnas_categoricas = [
        col for col in ["OverTime", "JobSatisfaction", "WorkLifeBalance", "MaritalStatus"]
        if col in datos.columns
    ]
    if columnas_categoricas:
        filas = int(np.ceil(len(columnas_categoricas) / 2))
        fig, ejes = plt.subplots(filas, 2, figsize=(12, 4 * filas))
        for eje, col in zip(np.asarray(ejes).reshape(-1), columnas_categoricas):
            sns.countplot(data=datos, x=col, hue=TARGET, palette="Set1", ax=eje)
            eje.set_title(f"{col} según rotación", fontweight='bold')
            eje.tick_params(axis="x", rotation=35)
        for eje in np.asarray(ejes).reshape(-1)[len(columnas_categoricas):]:
            eje.remove()
        fig.suptitle("Variables Categóricas vs Rotación", fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.show()

    # Mapa de correlación
    numericos = datos.select_dtypes(include=np.number).copy()
    numericos["Attrition_num"] = datos[TARGET].map({"No": 0, "Yes": 1})
    if numericos.shape[1] > 1:
        plt.figure(figsize=(14, 10))
        sns.heatmap(numericos.corr(), cmap="coolwarm", center=0, annot=False, linewidths=0.5)
        plt.title("Matriz de Correlación - Variables Numéricas", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Top correlaciones con Attrition
        correlaciones = numericos.corr()["Attrition_num"].drop("Attrition_num").sort_values()
        print("\nTop 10 correlaciones con Attrition:")
        print(correlaciones.head(10).to_string())


# ---------- PREPROCESAMIENTO Y MODELADO ----------
def crear_pipeline(columnas_numericas, columnas_categoricas, modelo, k_vecinos):
    pasos_antes_smote = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler()),
                ]),
                columnas_numericas,
            ),
            (
                "categoricas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="most_frequent")),
                    ("codificar", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                columnas_categoricas,
            ),
        ],
        remainder="drop",
    )

    indices_categoricos = list(
        range(len(columnas_numericas), len(columnas_numericas) + len(columnas_categoricas))
    )
    convertir_variables = ColumnTransformer(
        transformers=[
            ("numericas", "passthrough", list(range(len(columnas_numericas)))),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), indices_categoricos),
        ],
        remainder="drop",
    )

    return ImbalancedPipeline(
        steps=[
            ("preparar", pasos_antes_smote),
            ("smote", SMOTENC(categorical_features=indices_categoricos,
                              k_neighbors=k_vecinos, random_state=RANDOM_STATE)),
            ("variables", convertir_variables),
            ("clasificador", modelo),
        ]
    )


def preparar_variables(datos):
    X = datos.drop(columns=[TARGET]).copy()
    y = datos[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    # Eliminar columnas identificadoras/constantes
    columnas_identificadoras = [
        col for col in ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
        if col in X.columns
    ]
    X = X.drop(columns=columnas_identificadoras)

    constantes = [col for col in X if X[col].nunique(dropna=False) <= 1]
    X = X.drop(columns=constantes)

    columnas_numericas = X.select_dtypes(include=np.number).columns.tolist()
    columnas_categoricas = X.select_dtypes(exclude=np.number).columns.tolist()
    if not columnas_numericas and not columnas_categoricas:
        raise ValueError("No quedaron variables predictoras utilizables en el CSV.")
    return X, y, columnas_numericas, columnas_categoricas


def evaluar_modelos(X_train, X_test, y_train, y_test, columnas_numericas, columnas_categoricas):
    conteos = y_train.value_counts()
    menor_clase = int(conteos.min())
    if menor_clase < 2:
        raise ValueError("La clase menos frecuente tiene menos de 2 ejemplos.")
    k_vecinos = min(5, menor_clase - 1)
    print(f"\nConteo de entrenamiento antes de SMOTE: {conteos.to_dict()}")

    # Modelos a evaluar
    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    pipelines = {}
    predicciones = {}
    filas_resultados = []

    for nombre, modelo in modelos.items():
        print(f"\n>>> Entrenando: {nombre}...")
        pipeline = crear_pipeline(columnas_numericas, columnas_categoricas, modelo, k_vecinos)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        pipelines[nombre] = pipeline
        predicciones[nombre] = pred
        filas_resultados.append({
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
        })

    resultados = pd.DataFrame(filas_resultados).sort_values("F1-Score", ascending=False)
    print("\n" + "=" * 60)
    print("RESULTADOS COMPARATIVOS DE MODELOS")
    print("=" * 60)
    print(resultados.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Gráfica comparativa
    resultados_melt = resultados.melt(id_vars="Modelo",
                                       value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                                       var_name="Métrica", value_name="Valor")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=resultados_melt, x="Modelo", y="Valor", hue="Métrica", palette="viridis")
    plt.title("Comparación de Modelos por Métrica", fontsize=14, fontweight='bold')
    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)
    plt.legend(title="Métrica", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    mejor_modelo = resultados.iloc[0]["Modelo"]
    print(f"\n🏆 Mejor modelo por F1-Score: {mejor_modelo}")
    print("\nReporte de clasificación del mejor modelo:")
    print(classification_report(y_test, predicciones[mejor_modelo],
                                labels=[0, 1], target_names=["No rotación", "Rotación"],
                                zero_division=0))

    # Matriz de confusión
    matriz = confusion_matrix(y_test, predicciones[mejor_modelo], labels=[0, 1])
    plt.figure(figsize=(7, 6))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No rotación", "Rotación"],
                yticklabels=["No rotación", "Rotación"])
    plt.title(f"Matriz de Confusión - {mejor_modelo}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    plt.show()

    # Importancia de variables (Random Forest)
    pipeline_rf = pipelines["Random Forest"]
    nombres_variables = pipeline_rf.named_steps["variables"].get_feature_names_out()
    importancias = pipeline_rf.named_steps["clasificador"].feature_importances_
    top_variables = (
        pd.DataFrame({"Variable": nombres_variables, "Importancia": importancias})
        .nlargest(15, "Importancia")
        .sort_values("Importancia")
    )
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_variables, x="Importancia", y="Variable", hue="Variable", palette="magma", legend=False)
    plt.title("Top 15 Variables más Importantes - Random Forest", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return resultados


# ---------- FUNCIÓN PRINCIPAL ----------
def main():
    args = parse_args()
    print("=" * 60)
    print("PREDICCIÓN DE ROTACIÓN DE EMPLEADOS")
    print("=" * 60)

    datos = cargar_datos(args.csv)
    print(f"Archivo: {Path(args.csv).expanduser().resolve()}")
    print(f"Dimensiones: {datos.shape}")
    print(f"\nPrimeras filas:\n{datos.head().to_string(index=False)}")

    mostrar_analisis_exploratorio(datos)
    X, y, columnas_numericas, columnas_categoricas = preparar_variables(datos)

    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO Y MODELADO")
    print("=" * 60)
    print(f"Variables numéricas: {len(columnas_numericas)}")
    print(f"Variables categóricas: {len(columnas_categoricas)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    evaluar_modelos(X_train, X_test, y_train, y_test, columnas_numericas, columnas_categoricas)
    print("\n✅ Análisis completado con éxito.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"❌ Error: {error}") from error