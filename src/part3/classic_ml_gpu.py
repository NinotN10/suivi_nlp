#!/usr/bin/env python3
"""Partie 3 : Pipeline Machine Learning Classique GPU (RAPIDS & Optuna)."""

import os
import cudf
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.ensemble import RandomForestClassifier as cuRFClassifier
from cuml.svm import SVC as cuSVC
from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
from cuml.naive_bayes import GaussianNB as cuGaussianNB
import optuna
import json
import cupy as cp
from sklearn.metrics import classification_report

def load_data_gpu():
    """
    Charge train et test et concatène en un DataFrame cuDF.
    Convertit 'label' en binaire 0 (non-suicide) / 1 (suicide).
    """
    df_train = cudf.read_parquet("dataset/train.parquet")
    df_test = cudf.read_parquet("dataset/test.parquet")
    df = cudf.concat([df_train, df_test], ignore_index=True)
    df = df[["text", "label"]]
    df["label"] = (df["label"] == "suicide").astype("int32")
    return df

def main():
    os.makedirs("outputs/part3_ml", exist_ok=True)

    # 1) Chargement des données
    df = load_data_gpu()

    # 2) Split 70/10/20 train/val/test
    from cuml.model_selection import train_test_split
    df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val   = train_test_split(df_train_val, test_size=0.125, random_state=42)

    # 3) Optimisation pipeline Logistic Regression
    def objective_lr(trial):
        tfidf_max_features = trial.suggest_int("tfidf_max_features", 1000, 5000)
        tfidf_ngram_max = trial.suggest_int("tfidf_ngram_max", 1, 3)
        tfidf_ngram_range = (1, tfidf_ngram_max)
        tfidf_min_df       = trial.suggest_int("tfidf_min_df", 1, 5)
        C                  = trial.suggest_float("C", 1e-3, 1e3, log=True)
        penalty            = trial.suggest_categorical("penalty", ["l1","l2"])
        tol                = trial.suggest_float("tol", 1e-5, 1e-1, log=True)

        tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df
        )
        X_tr = tfidf.fit_transform(df_train["text"])
        X_val_t = tfidf.transform(df_val["text"])

        clf = cuLogisticRegression(C=C, penalty=penalty, tol=tol, solver="qn", max_iter=500)
        clf.fit(X_tr, df_train["label"])
        preds = clf.predict(X_val_t)
        return float((preds == df_val["label"]).mean())

    study_lr = optuna.create_study(direction="maximize")
    study_lr.optimize(objective_lr, n_trials=5)

    # 4) Optimisation pipeline Random Forest
    def objective_rf(trial):
        tfidf_max_features = trial.suggest_int("tfidf_max_features", 1000, 5000)
        tfidf_ngram_max = trial.suggest_int("tfidf_ngram_max", 1, 3)
        tfidf_ngram_range = (1, tfidf_ngram_max)
        tfidf_min_df       = trial.suggest_int("tfidf_min_df", 1, 5)
        n_estimators       = trial.suggest_int("n_estimators", 10, 200)
        max_depth          = trial.suggest_int("max_depth", 2, 20)

        tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df
        )
        X_tr = tfidf.fit_transform(df_train["text"])
        X_val_t = tfidf.transform(df_val["text"])

        X_tr_dense  = cp.asarray(X_tr.toarray())
        y_tr        = cp.asarray(df_train["label"].to_pandas().to_numpy())
        X_val_dense = cp.asarray(X_val_t.toarray())
        y_val       = cp.asarray(df_val["label"].to_pandas().to_numpy())

        rf = cuRFClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(X_tr_dense, y_tr)
        preds = rf.predict(X_val_dense)
        return float((preds == y_val).mean())

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=5)

    # 5) Optimisation pipeline K-Nearest Neighbors
    def objective_knn(trial):
        tfidf_max_features = trial.suggest_int("tfidf_max_features", 1000, 5000)
        tfidf_ngram_max = trial.suggest_int("tfidf_ngram_max", 1, 3)
        tfidf_ngram_range = (1, tfidf_ngram_max)
        tfidf_min_df       = trial.suggest_int("tfidf_min_df", 1, 5)
        n_neighbors        = trial.suggest_int("knn_n_neighbors", 1, 20)
        weights = "uniform"

        tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df
        )
        X_tr = tfidf.fit_transform(df_train["text"])
        X_val_t = tfidf.transform(df_val["text"])

        X_tr_dense  = cp.asarray(X_tr.toarray())
        y_tr        = cp.asarray(df_train["label"].to_pandas().to_numpy())
        X_val_dense = cp.asarray(X_val_t.toarray())
        y_val       = cp.asarray(df_val["label"].to_pandas().to_numpy())

        knn = cuKNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_tr_dense, y_tr)
        preds = knn.predict(X_val_dense)
        return float((preds == y_val).mean())

    study_knn = optuna.create_study(direction="maximize")
    study_knn.optimize(objective_knn, n_trials=5)

    # 6) Optimisation pipeline Support Vector Classifier
    def objective_svc(trial):
        tfidf_max_features = trial.suggest_int("tfidf_max_features", 500, 2000)
        tfidf_ngram_max    = trial.suggest_int("tfidf_ngram_max", 1, 3)
        tfidf_ngram_range  = (1, tfidf_ngram_max)
        tfidf_min_df       = trial.suggest_int("tfidf_min_df", 1, 5)
        C                   = trial.suggest_float("svc_C", 1e-3, 1e3, log=True)
        kernel              = trial.suggest_categorical("svc_kernel", ["linear","rbf"])
        gamma               = trial.suggest_float("svc_gamma", 1e-4, 1e-1, log=True)

        tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df
        )
        X_tr = tfidf.fit_transform(df_train["text"])
        X_val_t = tfidf.transform(df_val["text"])

        X_tr_dense  = cp.asarray(X_tr.toarray())
        y_tr        = cp.asarray(df_train["label"].to_pandas().to_numpy())
        X_val_dense = cp.asarray(X_val_t.toarray())
        y_val       = cp.asarray(df_val["label"].to_pandas().to_numpy())

        svc = cuSVC(C=C, kernel=kernel, gamma=gamma)
        svc.fit(X_tr_dense, y_tr)
        preds = svc.predict(X_val_dense)
        return float((preds == y_val).mean())

    study_svc = optuna.create_study(direction="maximize")
    study_svc.optimize(objective_svc, n_trials=5)

    # 7) Optimisation pipeline Gaussian Naive Bayes
    def objective_nb(trial):
        tfidf_max_features = trial.suggest_int("tfidf_max_features", 1000, 5000)
        tfidf_ngram_max = trial.suggest_int("tfidf_ngram_max", 1, 3)
        tfidf_ngram_range = (1, tfidf_ngram_max)
        tfidf_min_df       = trial.suggest_int("tfidf_min_df", 1, 5)
        var_smoothing      = trial.suggest_float("nb_var_smoothing", 1e-9, 1e-5, log=True)

        tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df
        )
        X_tr = tfidf.fit_transform(df_train["text"])
        X_val_t = tfidf.transform(df_val["text"])

        X_tr_dense  = cp.asarray(X_tr.toarray())
        y_tr        = cp.asarray(df_train["label"].to_pandas().to_numpy())
        X_val_dense = cp.asarray(X_val_t.toarray())
        y_val       = cp.asarray(df_val["label"].to_pandas().to_numpy())

        nb = cuGaussianNB(var_smoothing=var_smoothing)
        nb.fit(X_tr_dense, y_tr)
        preds = nb.predict(X_val_dense)
        return float((preds == y_val).mean())

    study_nb = optuna.create_study(direction="maximize")
    study_nb.optimize(objective_nb, n_trials=5)

    # 8) Réentraînement final sur train+val et évaluation sur test
    y_train_val = df_train_val["label"]
    y_test_cpu  = df_test["label"].to_pandas().to_numpy()

    # Logistic Regression final
    tfidf_lr = TfidfVectorizer(
        max_features=study_lr.best_params["tfidf_max_features"],
        ngram_range=(1, study_lr.best_params["tfidf_ngram_max"]),
        min_df=study_lr.best_params["tfidf_min_df"]
    )
    X_train_val_lr = tfidf_lr.fit_transform(df_train_val["text"])
    X_test_lr      = tfidf_lr.transform(df_test["text"])
    clf_lr = cuLogisticRegression(
        C=study_lr.best_params["C"],
        penalty=study_lr.best_params["penalty"],
        tol=study_lr.best_params["tol"],
        solver="qn", max_iter=500
    )
    clf_lr.fit(X_train_val_lr, y_train_val)
    preds_lr = clf_lr.predict(X_test_lr)
    preds_lr_cpu = cp.asnumpy(preds_lr)
    report_lr = classification_report(y_test_cpu, preds_lr_cpu, target_names=["non-suicide","suicide"])
    with open("outputs/part3_ml/report_logistic_regression.txt","w") as f:
        f.write(report_lr)
    with open("outputs/part3_ml/hyperparameters_logistic_regression.json","w") as f:
        json.dump(study_lr.best_params, f)

    # Random Forest final
    tfidf_rf = TfidfVectorizer(
        max_features=study_rf.best_params["tfidf_max_features"],
        ngram_range=(1, study_rf.best_params["tfidf_ngram_max"]),
        min_df=study_rf.best_params["tfidf_min_df"]
    )
    X_train_val_rf = tfidf_rf.fit_transform(df_train_val["text"])
    X_test_rf      = tfidf_rf.transform(df_test["text"])
    X_trf_dense    = cp.asarray(X_train_val_rf.toarray())
    y_trf          = cp.asarray(y_train_val.to_pandas().to_numpy())
    clf_rf = cuRFClassifier(
        n_estimators=study_rf.best_params["n_estimators"],
        max_depth=study_rf.best_params["max_depth"]
    )
    clf_rf.fit(X_trf_dense, y_trf)
    preds_rf = clf_rf.predict(cp.asarray(X_test_rf.toarray()))
    preds_rf_cpu = cp.asnumpy(preds_rf)
    report_rf = classification_report(y_test_cpu, preds_rf_cpu, target_names=["non-suicide","suicide"])
    with open("outputs/part3_ml/report_random_forest.txt","w") as f:
        f.write(report_rf)
    with open("outputs/part3_ml/hyperparameters_random_forest.json","w") as f:
        json.dump(study_rf.best_params, f)

    # K-Nearest Neighbors final
    tfidf_knn = TfidfVectorizer(
        max_features=study_knn.best_params["tfidf_max_features"],
        ngram_range=(1, study_knn.best_params["tfidf_ngram_max"]),
        min_df=study_knn.best_params["tfidf_min_df"]
    )
    X_train_val_knn = tfidf_knn.fit_transform(df_train_val["text"])
    X_test_knn      = tfidf_knn.transform(df_test["text"])
    X_train_val_knn_dense = cp.asarray(X_train_val_knn.toarray())
    X_test_knn_dense       = cp.asarray(X_test_knn.toarray())
    y_tr_knn               = cp.asarray(y_train_val.to_pandas().to_numpy())
    clf_knn = cuKNeighborsClassifier(
        n_neighbors=study_knn.best_params["knn_n_neighbors"]
    )
    clf_knn.fit(X_train_val_knn_dense, y_tr_knn)
    preds_knn = clf_knn.predict(X_test_knn_dense)
    preds_knn_cpu = cp.asnumpy(preds_knn)
    report_knn = classification_report(y_test_cpu, preds_knn_cpu, target_names=["non-suicide","suicide"])
    with open("outputs/part3_ml/report_k_neighbors.txt","w") as f:
        f.write(report_knn)
    with open("outputs/part3_ml/hyperparameters_k_neighbors.json","w") as f:
        json.dump(study_knn.best_params, f)

    # Support Vector Classifier final
    tfidf_svc = TfidfVectorizer(
        max_features=study_svc.best_params["tfidf_max_features"],
        ngram_range=(1, study_svc.best_params["tfidf_ngram_max"]),
        min_df=study_svc.best_params["tfidf_min_df"]
    )
    X_train_val_svc = tfidf_svc.fit_transform(df_train_val["text"])
    X_test_svc      = tfidf_svc.transform(df_test["text"])
    X_train_val_svc_dense = cp.asarray(X_train_val_svc.toarray())
    X_test_svc_dense       = cp.asarray(X_test_svc.toarray())
    y_tr_svc               = cp.asarray(y_train_val.to_pandas().to_numpy())
    clf_svc = cuSVC(
        C=study_svc.best_params["svc_C"],
        kernel=study_svc.best_params["svc_kernel"],
        gamma=study_svc.best_params["svc_gamma"]
    )
    clf_svc.fit(X_train_val_svc_dense, y_tr_svc)
    preds_svc = clf_svc.predict(X_test_svc_dense)
    preds_svc_cpu = cp.asnumpy(preds_svc)
    report_svc = classification_report(y_test_cpu, preds_svc_cpu, target_names=["non-suicide","suicide"])
    with open("outputs/part3_ml/report_support_vector_classifier.txt","w") as f:
        f.write(report_svc)
    with open("outputs/part3_ml/hyperparameters_support_vector_classifier.json","w") as f:
        json.dump(study_svc.best_params, f)

    # Gaussian Naive Bayes final
    tfidf_nb = TfidfVectorizer(
        max_features=study_nb.best_params["tfidf_max_features"],
        ngram_range=(1, study_nb.best_params["tfidf_ngram_max"]),
        min_df=study_nb.best_params["tfidf_min_df"]
    )
    X_train_val_nb = tfidf_nb.fit_transform(df_train_val["text"])
    X_test_nb      = tfidf_nb.transform(df_test["text"])
    X_train_val_nb_dense = cp.asarray(X_train_val_nb.toarray())
    X_test_nb_dense       = cp.asarray(X_test_nb.toarray())
    y_tr_nb               = cp.asarray(y_train_val.to_pandas().to_numpy())
    clf_nb = cuGaussianNB(var_smoothing=study_nb.best_params["nb_var_smoothing"])
    clf_nb.fit(X_train_val_nb_dense, y_tr_nb)
    preds_nb = clf_nb.predict(X_test_nb_dense)
    preds_nb_cpu = cp.asnumpy(preds_nb)
    report_nb = classification_report(y_test_cpu, preds_nb_cpu, target_names=["non-suicide","suicide"])
    with open("outputs/part3_ml/report_gaussian_naive_bayes.txt","w") as f:
        f.write(report_nb)
    with open("outputs/part3_ml/hyperparameters_gaussian_naive_bayes.json","w") as f:
        json.dump(study_nb.best_params, f)

    print("Partie 3 ML classique GPU optimisée avec 5 classifieurs terminée. Rapports dans outputs/part3_ml/")

if __name__ == "__main__":
    main()
