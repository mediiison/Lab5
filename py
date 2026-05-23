
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)

from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

# ── 1. Загрузка датасета ──────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=== Первые строки датасета ===")
print(df.head())
print(f"\nРазмер: {df.shape}")
print("\n=== Пропуски ===")
print(df.isnull().sum())

# ── 2. Предобработка данных ───────────────────────────────────

# Удаляем малоинформативные столбцы
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Заполнение пропусков
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Кодирование категориальных признаков
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])          # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])

print("\n=== Датасет после предобработки ===")
print(df.head())
print(f"\nПропуски после обработки: {df.isnull().sum().sum()}")

# ── 3. Разделение на обучающую и тестовую выборки ─────────────
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape}")
print(f"Тестовая выборка:  {X_test.shape}")

# ── 4. Обучение ансамблевых моделей ──────────────────────────

# --- 4.1 Бэггинг (Bagging) ---
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bagging.fit(X_train, y_train)

# --- 4.2 Случайный лес (Random Forest) ---
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

# --- 4.3 AdaBoost ---
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train)

# --- 4.4 Градиентный бустинг (Gradient Boosting) ---
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

# ── 5. Оценка качества моделей ────────────────────────────────

models = {
    'Bagging':            bagging,
    'Random Forest':      rf,
    'AdaBoost':           ada,
    'Gradient Boosting':  gb,
}

results = {}
print("\n=== Accuracy на тестовой выборке ===")
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  {name:<22}: {acc:.4f}")

# ── 6. Детальные отчёты ───────────────────────────────────────
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"Модель: {name}")
    print(classification_report(y_test, y_pred,
                                target_names=['Не выжил', 'Выжил']))

# ── 7. Визуализация результатов ───────────────────────────────

# --- 7.1 Сравнение Accuracy ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names  = list(results.keys())
scores = list(results.values())
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

bars = axes[0].barh(names, scores, color=colors, edgecolor='black', height=0.5)
axes[0].set_xlim(0.7, 0.95)
axes[0].set_xlabel('Accuracy', fontsize=12)
axes[0].set_title('Сравнение ансамблевых моделей', fontsize=13, fontweight='bold')
for bar, score in zip(bars, scores):
    axes[0].text(score + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{score:.4f}', va='center', fontsize=11)

# --- 7.2 Матрица ошибок для лучшей модели ---
best_name = max(results, key=results.get)
best_model = models[best_name]
y_pred_best = best_model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=['Не выжил', 'Выжил'],
    cmap='Blues',
    ax=axes[1]
)
axes[1].set_title(f'Матрица ошибок — {best_name}', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nЛучшая модель: {best_name} (Accuracy = {results[best_name]:.4f})")

# --- 7.3 Важность признаков для Random Forest ---
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Важность признаков — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
