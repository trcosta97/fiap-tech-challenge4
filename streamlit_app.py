import os
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Tech Challenge 4 – Obesidade", page_icon="🍏", layout="wide")

@st.cache_data(show_spinner=False)
def load_dataset(upload):
    
    if upload is not None:
        return pd.read_csv(upload, index_col=0)
    default_path = os.path.join("DATASETS", "dados_machine_learning.csv")
    if os.path.exists(default_path):
        return pd.read_csv(default_path, index_col=0)
    raise FileNotFoundError("Nenhum arquivo encontrado. Faça upload de um CSV ou adicione DATASETS/dados_machine_learning.csv ao repositório.")

def prepare_data(df):
    df = df.copy()
    df["obesidade_binary"] = df["obesidade"].apply(lambda x: 0 if x <= 3 else 1)
    features_basicas = [
        "genero",
        "idade",
        "historico_familiar",
        "frequencia_consumo_alimentos_caloricos",
        "frequencia_consumo_vegetais",
        "numero_refeicoes",
        "consumo_lanches_entre_refeicoes",
        "fuma",
        "CH2O",
        "monitoramento_calorias",
        "frequencia_atividade_fisica",
        "tempo_diario_uso_dispositivos_eletronicos",
        "consumo_alcool",
        "tipo_transporte",
    ]
    X = df[features_basicas]
    y = df["obesidade_binary"]
    return X, y, features_basicas

def train_models(X, y, test_size=0.2, random_state=42):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "report": classification_report(y_test, rf_pred, target_names=["Não Obeso", "Obeso"]),
        "cm": confusion_matrix(y_test, rf_pred),
    }
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    results["Gradient Boosting"] = {
        "accuracy": accuracy_score(y_test, gb_pred),
        "report": classification_report(y_test, gb_pred, target_names=["Não Obeso", "Obeso"]),
        "cm": confusion_matrix(y_test, gb_pred),
    }
    # SVM
    svm_model = SVC(kernel="rbf", random_state=random_state)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    results["SVM"] = {
        "accuracy": accuracy_score(y_test, svm_pred),
        "report": classification_report(y_test, svm_pred, target_names=["Não Obeso", "Obeso"]),
        "cm": confusion_matrix(y_test, svm_pred),
    }
    return results

def plot_confusions(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res["cm"], annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Não Obeso", "Obeso"], yticklabels=["Não Obeso", "Obeso"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
    st.pyplot(fig)

def main():
    st.title("Tech Challenge 4: Predição de Obesidade")
    st.markdown(
        "Use o upload abaixo para carregar um CSV ou deixe em branco para usar o dataset padrão."
    )
    upload = st.file_uploader("Upload de CSV (opcional)", type=["csv"])
    try:
        df = load_dataset(upload)
    except Exception as e:
        st.error(str(e))
        return

    st.subheader("Prévia do dataset")
    st.dataframe(df.head(20), use_container_width=True)
    X, y, features = prepare_data(df)
    st.write(f"Total de registros: {len(df)}, Variáveis usadas: {len(features)}")
    st.write("Distribuição da variável alvo (0=Não obeso, 1=Obeso):")
    st.table(y.value_counts().rename_axis("Classe").reset_index(name="Contagem"))

    # Controles no sidebar
    test_size = st.sidebar.slider("Proporção da base de teste", 0.1, 0.5, 0.2, step=0.05)
    random_state = st.sidebar.number_input("Random state", value=42, step=1)

    if st.button("Treinar modelos", type="primary"):
        with st.spinner("Treinando..."):
            results = train_models(X, y, test_size=test_size, random_state=random_state)
        # Exibir acurácia
        st.subheader("Acurácia dos modelos")
        st.table(pd.DataFrame({
            "Modelo": list(results.keys()),
            "Acurácia": [res["accuracy"] for res in results.values()]
        }).sort_values("Acurácia", ascending=False))
        # Relatórios
        st.subheader("Relatórios de classificação")
        for name, res in results.items():
            st.markdown(f"**{name}**")
            st.text(res["report"])
        # Confusão
        st.subheader("Matrizes de confusão")
        plot_confusions(results)

if __name__ == "__main__":
    main()
