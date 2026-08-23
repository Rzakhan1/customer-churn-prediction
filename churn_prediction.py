import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

def generate_customer_data(n_customers=2500):
    rng = np.random.default_rng(RANDOM_STATE)
    tenure_months = rng.integers(1, 73, n_customers)
    monthly_charges = np.clip(rng.normal(70, 25, n_customers), 20, 130)
    support_tickets = rng.poisson(1.5, n_customers)
    late_payments = rng.poisson(0.8, n_customers)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_customers, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(
        ["DSL", "Fiber", "None"],
        size=n_customers, p=[0.35, 0.55, 0.10]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_customers, p=[0.65, 0.35])

    total_charges = np.clip(
        tenure_months * monthly_charges + rng.normal(0, 100, n_customers),
        0, None
    )

    logit = (
        -1.8
        + 1.2 * (contract_type == "Month-to-month")
        - 0.9 * (contract_type == "Two year")
        + 0.018 * (monthly_charges - 70)
        - 0.025 * (tenure_months - 24)
        + 0.28 * support_tickets
        + 0.35 * late_payments
        + 0.55 * (internet_service == "Fiber")
        + 0.20 * (paperless_billing == "Yes")
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "paperless_billing": paperless_billing,
        "churn": churn
    })

def build_preprocessor():
    numeric_features = [
        "tenure_months", "monthly_charges", "total_charges",
        "support_tickets", "late_payments"
    ]
    categorical_features = [
        "contract_type", "internet_service", "paperless_billing"
    ]

    return ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

def evaluate_model(name, pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities)
    }

    print("\n" + name)
    print(classification_report(y_test, predictions, digits=3))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions)
    ).plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(name.lower().replace(" ", "_") + "_confusion_matrix.png", dpi=150)
    plt.close()

    return metrics

def main():
    df = generate_customer_data()
    df.to_csv("customer_churn.csv", index=False)

    X = df.drop(columns="churn")
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=4,
                class_weight="balanced", random_state=RANDOM_STATE
            ))
        ])
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test))

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv("model_results.csv")
    print("\nModel comparison:")
    print(results_df.round(3))

    rf = models["Random Forest"]
    feature_names = rf.named_steps["preprocessor"].get_feature_names_out()
    importances = rf.named_steps["model"].feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.xlabel("Feature Importance")
    plt.title("Top Random Forest Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
