# src/train.py
import warnings
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# Persistent MLflow directory (host machine)
mlflow.set_tracking_uri("file:///app/mlruns")
mlflow.set_experiment("Credit_Risk_Model_Experiments")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class CreditRiskTrainer:
    def __init__(self, X, y, test_size=0.2, random_state=42, experiment_name="Credit_Risk_Model_Experiments"):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.models = {}
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def add_model(self, name, model, param_grid=None):
        """Add a model and optional parameter grid for tuning."""
        self.models[name] = {"model": model, "param_grid": param_grid}

    def train_and_tune(self):
        """Train all added models and optionally tune with GridSearchCV."""
        for name, info in self.models.items():
            model = info["model"]
            param_grid = info["param_grid"]

            print(f"\nTraining model: {name}")

            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
                gs.fit(self.X_train, self.y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(self.X_train, self.y_train)
                best_model = model

            self.models[name]["best_model"] = best_model

            # Evaluate
            y_pred = best_model.predict(self.X_test)
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "f1_score": f1_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, best_model.predict_proba(self.X_test)[:, 1])
            }

            self.models[name]["metrics"] = metrics
            print(f"Metrics for {name}: {metrics}")

    def log_experiments(self):
        """Log all trained models and metrics to MLflow."""
        for name, info in self.models.items():
            best_model = info["best_model"]
            metrics = info["metrics"]

            with mlflow.start_run(run_name=name):
                mlflow.log_params(best_model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best_model, artifact_path=name)
                print(f"Logged {name} to MLflow with metrics: {metrics}")

    def get_best_model(self, metric="roc_auc"):
        """Return the best model based on a given metric."""
        best_name = max(
            self.models,
            key=lambda x: self.models[x]["metrics"][metric]
        )
        return best_name, self.models[best_name]["best_model"], self.models[best_name]["metrics"][metric]

    def save_best_model_locally(self, path="notebooks/models/best_model"):
        """Save the best model to a local path."""
        best_name, best_model, best_metric = self.get_best_model()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mlflow.sklearn.save_model(best_model, path)
        print(f"Saved best model '{best_name}' locally at: {path} (metric={best_metric:.4f})")

    def register_best_model(self, metric="roc_auc", registered_model_name="CreditRiskModel"):
        """Register the best model in MLflow Model Registry."""
        best_name, best_model, best_metric = self.get_best_model(metric)

        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=f"tags.mlflow.runName = '{best_name}'",
            order_by=[f"metrics.{metric} DESC"]
        )

        if runs.empty:
            raise RuntimeError("No MLflow run found for best model")

        run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/{best_name}"

        mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )

        print(f"Registered '{best_name}' as '{registered_model_name}' (metric={best_metric:.4f})")
