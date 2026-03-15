import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


def load_iris_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Ulazni fajl je prazan.")
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    preferred_names = ["species", "class", "target", "label", "y"]
    lower_to_original = {col.lower(): col for col in df.columns}

    for name in preferred_names:
        if name in lower_to_original:
            return lower_to_original[name]

    return df.columns[-1]


def build_parameter_grid() -> dict:
    return {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 2, 3, 4, 5, 6],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 3, 4],
    }


def print_cv_results(results_df: pd.DataFrame) -> None:
    print("\n=== REZULTATI ZA SVE KOMBINACIJE PARAMETARA (CV) ===")
    display_cols = [
        "param_criterion",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "mean_test_precision_macro",
        "std_test_precision_macro",
        "mean_test_f1_macro",
        "std_test_f1_macro",
    ]

    results_sorted = results_df.sort_values(
        by=["mean_test_precision_macro", "mean_test_f1_macro"],
        ascending=False,
    )

    for _, row in results_sorted[display_cols].iterrows():
        print(
            "params="
            f"{{criterion={row['param_criterion']}, "
            f"max_depth={row['param_max_depth']}, "
            f"min_samples_split={row['param_min_samples_split']}, "
            f"min_samples_leaf={row['param_min_samples_leaf']}}} | "
            f"precision_macro: mean={row['mean_test_precision_macro']:.4f}, "
            f"std={row['std_test_precision_macro']:.4f} | "
            f"f1_macro: mean={row['mean_test_f1_macro']:.4f}, "
            f"std={row['std_test_f1_macro']:.4f}"
        )


def get_best_params_for_metric(results_df: pd.DataFrame, metric_name: str) -> dict:
    best_idx = results_df[f"mean_test_{metric_name}"].idxmax()
    return results_df.loc[best_idx, "params"]


def fit_and_report(
    metric_name: str,
    best_params: dict,
    x_train: pd.DataFrame,
    y_train,
    x_test: pd.DataFrame,
    y_test,
    target_names,
) -> None:
    model = DecisionTreeClassifier(random_state=42, **best_params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"\n=== NAJBOLJI MODEL PO MERI: {metric_name} ===")
    print(f"Najbolji parametri: {best_params}")
    print("Classification report (test skup):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )


def main() -> None:
    data_path = "iris.csv"

    df = load_iris_dataframe(data_path)
    target_col = detect_target_column(df)

    x = df.drop(columns=[target_col])
    y_raw = df[target_col]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    target_names = label_encoder.classes_.astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    param_grid = build_parameter_grid()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring={"precision_macro": "precision_macro", "f1_macro": "f1_macro"},
        refit=False,
        cv=cv,
        n_jobs=-1,
    )
    gs.fit(x_train, y_train)

    results_df = pd.DataFrame(gs.cv_results_)
    print_cv_results(results_df)

    best_precision_params = get_best_params_for_metric(results_df, "precision_macro")
    best_f1_params = get_best_params_for_metric(results_df, "f1_macro")

    fit_and_report(
        metric_name="precision_macro",
        best_params=best_precision_params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        target_names=target_names,
    )

    fit_and_report(
        metric_name="f1_macro",
        best_params=best_f1_params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        target_names=target_names,
    )


if __name__ == "__main__":
    main()
