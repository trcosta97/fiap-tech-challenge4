import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração para gráficos em português
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_prepare_data():
    """Carrega e prepara os dados"""
    df = pd.read_csv('../DATASETS/dados_machine_learning.csv', index_col=0)
    
    # Target binário
    df['obesidade_binary'] = df['obesidade'].apply(lambda x: 0 if x <= 3 else 1)
    
    # Features sem data leakage
    features_basicas = ['genero', 'idade', 'historico_familiar', 'frequencia_consumo_alimentos_caloricos',
                       'frequencia_consumo_vegetais', 'numero_refeicoes', 'consumo_lanches_entre_refeicoes',
                       'fuma', 'CH2O', 'monitoramento_calorias', 'frequencia_atividade_fisica',
                       'tempo_diario_uso_dispositivos_eletronicos', 'consumo_alcool', 'tipo_transporte']
    
    X = df[features_basicas]
    y = df['obesidade_binary']
    
    return X, y, features_basicas

def train_models(X, y):
    """Treina os modelos e retorna resultados"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalização para SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos
    models = {}
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    models['Random Forest'] = {
        'model': rf_model,
        'accuracy': accuracy_score(y_test, rf_pred),
        'predictions': rf_pred
    }
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    models['Gradient Boosting'] = {
        'model': gb_model,
        'accuracy': accuracy_score(y_test, gb_pred),
        'predictions': gb_pred
    }
    
    # SVM
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    models['SVM'] = {
        'model': svm_model,
        'accuracy': accuracy_score(y_test, svm_pred),
        'predictions': svm_pred
    }
    
    return models, y_test

def create_feature_importance_plot(best_model, feature_names):
    """Cria gráfico de importância das features"""
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        
        # Nomes mais legíveis
        feature_labels = {
            'genero': 'Gênero',
            'idade': 'Idade',
            'historico_familiar': 'Histórico Familiar',
            'frequencia_consumo_alimentos_caloricos': 'Consumo Alimentos Calóricos',
            'frequencia_consumo_vegetais': 'Consumo de Vegetais',
            'numero_refeicoes': 'Número de Refeições',
            'consumo_lanches_entre_refeicoes': 'Lanches Entre Refeições',
            'fuma': 'Fumante',
            'CH2O': 'Consumo de Água',
            'monitoramento_calorias': 'Monitora Calorias',
            'frequencia_atividade_fisica': 'Atividade Física',
            'tempo_diario_uso_dispositivos_eletronicos': 'Tempo Tela/Dia',
            'consumo_alcool': 'Consumo de Álcool',
            'tipo_transporte': 'Tipo de Transporte'
        }
        
        # Ordenar por importância
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Importância das Features - Random Forest', fontsize=16, fontweight='bold')
        
        # Criar barras
        bars = plt.bar(range(len(importances)), importances[indices], color='skyblue', alpha=0.8)
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Labels
        plt.xticks(range(len(importances)), 
                  [feature_labels.get(feature_names[i], feature_names[i]) for i in indices], 
                  rotation=45, ha='right')
        plt.ylabel('Importância', fontsize=12)
        plt.xlabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importances[indices], [feature_names[i] for i in indices]

def create_least_important_features_plot(best_model, feature_names):
    """Cria gráfico das 5 features menos importantes"""
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        
        # Nomes mais legíveis
        feature_labels = {
            'genero': 'Gênero',
            'idade': 'Idade',
            'historico_familiar': 'Histórico Familiar',
            'frequencia_consumo_alimentos_caloricos': 'Consumo Alimentos Calóricos',
            'frequencia_consumo_vegetais': 'Consumo de Vegetais',
            'numero_refeicoes': 'Número de Refeições',
            'consumo_lanches_entre_refeicoes': 'Lanches Entre Refeições',
            'fuma': 'Fumante',
            'CH2O': 'Consumo de Água',
            'monitoramento_calorias': 'Monitora Calorias',
            'frequencia_atividade_fisica': 'Atividade Física',
            'tempo_diario_uso_dispositivos_eletronicos': 'Tempo Tela/Dia',
            'consumo_alcool': 'Consumo de Álcool',
            'tipo_transporte': 'Tipo de Transporte'
        }
        
        # Ordenar por importância (menor para maior)
        indices = np.argsort(importances)
        
        # Pegar apenas as 5 menos importantes
        least_important_indices = indices[:5]
        least_importances = importances[least_important_indices]
        
        plt.figure(figsize=(10, 6))
        plt.title('5 Features Menos Importantes - Random Forest', fontsize=16, fontweight='bold')
        
        # Criar barras com cor diferente (vermelho claro)
        bars = plt.bar(range(len(least_importances)), least_importances, color='lightcoral', alpha=0.8)
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Labels
        plt.xticks(range(len(least_importances)), 
                  [feature_labels.get(feature_names[i], feature_names[i]) for i in least_important_indices], 
                  rotation=45, ha='right')
        plt.ylabel('Importância', fontsize=12)
        plt.xlabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('least_important_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return least_importances, [feature_names[i] for i in least_important_indices]

def create_model_comparison_plot(models):
    """Cria gráfico de comparação dos modelos"""
    model_names = list(models.keys())
    accuracies = [models[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['lightcoral', 'lightblue', 'lightgreen'], alpha=0.8)
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Comparação de Acurácia dos Modelos', fontsize=16, fontweight='bold')
    plt.ylabel('Acurácia', fontsize=12)
    plt.xlabel('Modelos', fontsize=12)
    plt.ylim(0, 1)
    
    # Linha de referência 75%
    plt.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Meta: 75%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Gerando materiais para apresentação...")
    
    # Carregar dados
    X, y, feature_names = load_and_prepare_data()
    print(f"✅ Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Treinar modelos
    models, y_test = train_models(X, y)
    print("✅ Modelos treinados")
    
    # Encontrar melhor modelo
    best_model_name = max(models, key=lambda x: models[x]['accuracy'])
    best_model = models[best_model_name]['model']
    best_accuracy = models[best_model_name]['accuracy']
    
    print(f"🏆 Melhor modelo: {best_model_name} ({best_accuracy:.1%})")
    
    # Gerar gráficos
    print("📊 Gerando gráfico de comparação...")
    create_model_comparison_plot(models)
    
    print("📊 Gerando gráfico de feature importance...")
    importances, sorted_features = create_feature_importance_plot(best_model, feature_names)
    
    print("📊 Gerando gráfico de features menos importantes...")
    least_importances, least_features = create_least_important_features_plot(best_model, feature_names)
    
    # Resumo dos resultados
    print("\n" + "="*50)
    print("📋 RESUMO DOS RESULTADOS")
    print("="*50)
    for name, data in models.items():
        print(f"{name}: {data['accuracy']:.1%}")
    
    print(f"\n🎯 Meta de 75%: {'✅ ATINGIDA' if best_accuracy >= 0.75 else '❌ NÃO ATINGIDA'}")
    
    print(f"\n🔝 Top 5 Features mais importantes:")
    for i in range(min(5, len(sorted_features))):
        feature_name = sorted_features[i]
        importance = importances[i]
        print(f"  {i+1}. {feature_name}: {importance:.3f}")
    
    print(f"\n🔻 5 Features menos importantes:")
    for i in range(len(least_features)):
        feature_name = least_features[i]
        importance = least_importances[i]
        print(f"  {i+1}. {feature_name}: {importance:.3f}")
    
    print("\n✅ Gráficos salvos: feature_importance.png, model_comparison.png e least_important_features.png")

if __name__ == "__main__":
    main()