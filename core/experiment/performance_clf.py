from sklearn.metrics import *
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    roc_curve,
)
from matplotlib.colors import ListedColormap

from utils.python.printing import Printing
from utils.python.display import display_confusion_matrix

import numpy as np
import pandas as pd

class ClassificationPerformance(Printing): 
    def __init__(
        self,

        # Context Object
        context = None,

        # Dataset
        y = None, 
        y_hat = None,
        y_proba = None
    ):
        # Context 
        self.split_run = context

        # Score Containers 
        self.scores = {
            "overall" : {},
            "binary" : {},
            "micro" : {}, 
            "macro" : {},
            "weighted" : {},
            "per_class" : {}
        }

        # Dataset 
        self.y = y 
        self.y_hat = y_hat 
        self.y_proba = y_proba

    def compute(self): 
        self.print(f"> Computing performance.")
        self.compute_accuracy_score() 
        self.compute_balanced_accuracy_score() 
        self.compute_cohens_kappa_score()
        self.compute_matthews_correlation_coefficient_score() 
        self.compute_f1_score() 
        self.compute_precision_score()
        self.compute_recall_score() 
        self.compute_pr_auc_score() 
        self.compute_roc_auc_score() 
        self.compute_brier_score_loss() 
        self.compute_hinge_loss() 
        self.compute_log_loss() 
        self.generate_confusion_matrix() 
        self.generate_pr_auc_plot() 
        self.generate_roc_auc_plot()

        self.generate_decision_boundary_display(
            self.split_run.X_train, self.split_run.y_train
        ) 
        self.generate_decision_boundary_display(
            self.split_run.X_test, self.split_run.y_test
        ) 

    def per_class_scores(self, scoring, y, y_hat): 
        results = {} 
        classes_ = self.split_run.experiment.classes 
        for class_ in classes_: 
            self.print(f"\t\t:: Computing OVR for {class_}.")
            y_ = y == class_
            y_hat_ = y_hat == class_
            score = scoring(y_, y_hat_)
            results[class_] = score
            self.print(f"\t\t\t:: Result: {score}")
    
    def per_class_proba_scores(
        self, scoring, y, y_hat, weighted=True, *args, **kwargs
    ): 
        results = {} 
        classes_ = self.split_run.experiment.classes 
        i = 0
        for class_ in classes_: 
            self.print(f"\t\t:: Computing OVR for {class_}.")
            y_ = y == class_
            y_hat_ = y_hat[:, i]
            score = scoring(y_, y_hat_)
            results[class_] = score
            self.print(f"\t\t\t:: Result: {score}")
            i += 1
        
    def compute_accuracy_score(self): 
        self.print(f"> Computing accuracy score.")

        # Overall Score
        self.print(f"\t:: Computing overall accuracy score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = accuracy_score(
            self.y, self.y_hat, sample_weight=sample_weights
        )
        self.scores["overall"]["accuracy"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_scores(
            lambda y, y_hat: accuracy_score(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), self.y, self.y_hat
        )
        self.scores["per_class"]["accuracy"] = scores
        self.print(f"\t\t:: Result: {score}")

    def compute_balanced_accuracy_score(self):
        self.print(f"> Computing Balanced Accuracy Score.")

        # Overall Score
        self.print(f"\t:: Computing overall score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = balanced_accuracy_score(
            self.y, self.y_hat, sample_weight=sample_weights
        )
        self.scores["overall"]["balanced_accuracy"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_scores(
            lambda y, y_hat: balanced_accuracy_score(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), self.y, self.y_hat
        )
        self.scores["per_class"]["balanced_accuracy"] = scores
        self.print(f"\t\t:: Result: {score}")

    def compute_cohens_kappa_score(self): 
        self.print(f"> Computing Cohen's Kappa score.") 

        # Overall Score
        self.print(f"\t:: Computing overall score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = cohen_kappa_score(self.y, self.y_hat, sample_weight=sample_weights)
        self.scores["overall"]["cohens_kappa"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_scores(
            lambda y, y_hat: cohen_kappa_score(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), self.y, self.y_hat
        )
        self.scores["per_class"]["cohens_kappa"] = scores
        self.print(f"\t\t:: Result: {score}")

    def compute_matthews_correlation_coefficient_score(self): 
        self.print(f"> Computing Mathew's Correlation Coefficient.")

        # Overall Score
        self.print(f"\t:: Computing overall score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = matthews_corrcoef(self.y, self.y_hat, sample_weight=sample_weights)
        self.scores["overall"]["cohens_kappa"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_scores(
            lambda y, y_hat: matthews_corrcoef(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), self.y, self.y_hat
        )
        self.scores["per_class"]["cohens_kappa"] = scores
        self.print(f"\t\t:: Result: {score}")


    def compute_f1_score(self): 
        self.print(f"> Computing F1 score.")

        # Micro Score
        self.print(f"\t:: Computing micro score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = f1_score(
            self.y, self.y_hat, average="micro", sample_weight=sample_weights
        )
        self.scores["micro"]["f1_score"] = score
        self.print(f"\t\t:: Result: {score}")

        # Macro Score
        self.print(f"\t:: Computing macro score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = f1_score(
            self.y, self.y_hat, average="macro", sample_weight=sample_weights
        )
        self.scores["macro"]["f1_score"] = score
        self.print(f"\t\t:: Result: {score}")

        # Weighted Score
        self.print(f"\t:: Computing weighted score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = f1_score(
            self.y, self.y_hat, average="weighted", sample_weight=sample_weights
        )
        self.scores["weighted"]["f1_score"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_scores(
            lambda y, y_hat: average_precision_score(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), self.y, self.y_hat
        )
        self.scores["per_class"]["f1_score"] = scores
        self.print(f"\t\t:: Result: {score}")

    def compute_precision_score(self): 
        self.print(f"> Computing precision score.") 

        # Micro Score
        self.print(f"\t:: Computing micro score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = precision_score(
            self.y, self.y_hat, average="micro", sample_weight=sample_weights
        )
        self.scores["micro"]["precision"] = score
        self.print(f"\t\t:: Result: {score}")

        # Macro Score
        self.print(f"\t:: Computing macro score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = precision_score(
            self.y, self.y_hat, average="macro", sample_weight=sample_weights
        )
        self.scores["macro"]["precision"] = score
        self.print(f"\t\t:: Result: {score}")

        # Weighted Score
        self.print(f"\t:: Computing weighted score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = precision_score(
            self.y, self.y_hat, average="weighted", sample_weight=sample_weights
        )
        self.scores["weighted"]["precision"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        scores = self.per_class_scores(
            lambda y, y_hat: average_precision_score(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), self.y, self.y_hat
        )
        self.scores["per_class"]["precision"] = scores
        self.print(f"\t\t:: Result: {score}")

    
    def compute_recall_score(self): 
        self.print(f"> Computing recall score.") 

        # Micro Score
        self.print(f"\t:: Computing micro score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = precision_score(
            self.y, self.y_hat, average="micro", sample_weight=sample_weights
        )
        self.scores["micro"]["recall"] = score
        self.print(f"\t\t:: Result: {score}")

        # Macro Score
        self.print(f"\t:: Computing macro score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = precision_score(
            self.y, self.y_hat, average="macro", sample_weight=sample_weights
        )
        self.scores["macro"]["recall"] = score
        self.print(f"\t\t:: Result: {score}")

        # Weighted Score
        self.print(f"\t:: Computing weighted score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = precision_score(
            self.y, self.y_hat, average="weighted", sample_weight=sample_weights
        )
        self.scores["weighted"]["recall"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_scores(
            lambda y, y_hat: precision_score(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), self.y, self.y_hat
        )
        self.scores["per_class"]["recall"] = scores
        self.print(f"\t\t:: Result: {score}")
       
    def compute_pr_auc_score(self): 
        self.print(f"> Computing PR-AUC score.")

        if self.split_run.experiment.multiclass:
            # Micro Score
            self.print(f"\t:: Computing micro score.")
            score = average_precision_score(self.y, self.y_proba, average="micro")
            self.scores["micro"]["pr_auc"] = score
            self.print(f"\t\t:: Result: {score}")

            # Macro Score
            self.print(f"\t:: Computing macro score.")
            score = average_precision_score(self.y, self.y_proba, average="macro")
            self.scores["macro"]["pr_auc"] = score
            self.print(f"\t\t:: Result: {score}")

            # Weighted Score
            self.print(f"\t:: Computing weighted score.")
            score = average_precision_score(self.y, self.y_proba, average="weighted")
            self.scores["weighted"]["pr_auc"] = score
            self.print(f"\t\t:: Result: {score}")

            # Per-Class Score
            self.print(f"\t:: Computing per-class score.")
            scores = self.per_class_proba_scores(
                average_precision_score, self.y, self.y_proba
            )
            self.scores["per_class"]["pr_auc"] = scores
            self.print(f"\t\t:: Result: {score}")
        else: 
            # Per-Class Score
            self.print(f"\t:: Computing per-class score.")
            scores = self.per_class_scores(
                average_precision_score, 
                self.y, 
                self.y_hat
            )
            self.scores["per_class"]["pr_auc"] = scores
            self.print(f"\t\t:: Result: {scores}")

    def compute_roc_auc_score(self): 
        self.print(f"> Computing ROC-AUC score.")

        if self.split_run.experiment.multiclass:
            # Micro Score
            self.print(f"\t:: Computing micro score.")
            score = average_precision_score(self.y, self.y_proba, average="micro")
            self.scores["micro"]["roc_auc"] = score
            self.print(f"\t\t:: Result: {score}")

            # Macro Score
            self.print(f"\t:: Computing macro score.")
            score = average_precision_score(self.y, self.y_proba, average="macro")
            self.scores["macro"]["roc_auc"] = score
            self.print(f"\t\t:: Result: {score}")

            # Weighted Score
            self.print(f"\t:: Computing weighted score.")
            score = average_precision_score(self.y, self.y_proba, average="weighted")
            self.scores["weighted"]["roc_auc"] = score
            self.print(f"\t\t:: Result: {score}")

            # Per-Class Score
            self.print(f"\t:: Computing per-class score.")
            scores = self.per_class_proba_scores(
                average_precision_score, self.y, self.y_proba
            )
            self.scores["per_class"]["roc_auc"] = scores
            self.print(f"\t\t:: Result: {score}")
        else: 
            # Per-Class Score
            self.print(f"\t:: Computing per-class score.")
            scores = self.per_class_proba_scores(
                average_precision_score, 
                self.y, 
                self.y_proba
            )
            self.scores["per_class"]["roc_auc"] = scores
            self.print(f"\t\t:: Result: {scores}")

    def compute_brier_score_loss(self): 
        self.print(f"> Compute Brier Score Loss.")
        
        # Overall Score
        self.print(f"\t:: Computing score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = brier_score_loss(self.y, self.y_hat, sample_weight=sample_weights)
        self.scores["overall"]["brier_score_loss"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_proba_scores(
            lambda y, y_hat: average_precision_score(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), 
            self.y, 
            self.y_proba
        )
        self.scores["per_class"]["brier_score_loss"] = scores
        self.print(f"\t\t:: Result: {scores}")

    def compute_hinge_loss(self):
        self.print(f"> Compute Hinge Loss.")

        # Overall Score
        self.print(f"\t:: Computing overall score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = hinge_loss(self.y, self.y_hat, sample_weight=sample_weights)
        self.scores["overall"]["hinge_loss"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_proba_scores(
            lambda y, y_hat: hinge_loss(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), 
            self.y, 
            self.y_proba
        )
        self.scores["per_class"]["hinge_loss"] = scores
        self.print(f"\t\t:: Result: {scores}")
        
    
    def compute_log_loss(self): 
        self.print("> Computing Log Loss.")

        # Overall Score
        self.print(f"\t:: Computing overall score.")
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y)
        score = log_loss(self.y, self.y_hat, sample_weight=sample_weights)
        self.scores["overall"]["log_loss"] = score
        self.print(f"\t\t:: Result: {score}")

        # Per-Class Score
        self.print(f"\t:: Computing per-class score.")
        scores = self.per_class_proba_scores(
            lambda y, y_hat: log_loss(
                y, y_hat, sample_weight=compute_sample_weight(
                    class_weight='balanced',
                    y=y
                )
            ), 
            self.y, 
            self.y_proba
        )
        self.scores["per_class"]["roc_auc"] = scores
        self.print(f"\t\t:: Result: {scores}")

    def generate_confusion_matrix(self): 
        self.print("> Computing Confusion Matrix.")
        cm = confusion_matrix(
            self.y, self.y_hat
        )
        self.scores["confusion_matrix"] = cm
        labels = self.split_run.experiment.classes
        display_confusion_matrix(cm, labels, prefix=self.indent + "\t")

        fig, ax = plt.subplots(figsize=(6, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.split_run.experiment.classes
        )
        disp.plot(ax=ax, colorbar=False)

        

    def generate_pr_auc_plot(self):
        self.print("> Generating PR-AUC Plot.")

        fig, ax = plt.subplots(figsize=(6, 6))

        classes = self.split_run.experiment.classes

        if len(classes) == 2:

            proba = self.y_proba[:, 1] if self.y_proba.ndim == 2 else self.y_proba

            PrecisionRecallDisplay.from_predictions(
                self.y,
                proba,
                ax=ax,
                name=str(classes[1])
            )

        else:

            y_bin = label_binarize(self.y, classes=classes)

            for i, cls in enumerate(classes):
                PrecisionRecallDisplay.from_predictions(
                    y_bin[:, i],
                    self.y_proba[:, i],
                    ax=ax,
                    name=str(cls)
                )

        ax.set_title("Precision-Recall Curve")
        fig.tight_layout()

        plt.show()


        self.pr_auc_plot = (fig, ax)

    def generate_roc_auc_plot(self):
        self.print("> Generating ROC-AUC Plot.")

        fig, ax = plt.subplots(figsize=(6, 6))

        classes = self.split_run.experiment.classes

        if len(classes) == 2:

            proba = self.y_proba[:, 1] if self.y_proba.ndim == 2 else self.y_proba

            RocCurveDisplay.from_predictions(
                self.y,
                proba,
                ax=ax,
                name=str(classes[1])
            )

        else:

            y_bin = label_binarize(self.y, classes=classes)

            for i, cls in enumerate(classes):
                RocCurveDisplay.from_predictions(
                    y_bin[:, i],
                    self.y_proba[:, i],
                    ax=ax,
                    name=str(cls)
                )

        ax.set_title("ROC Curve")
        fig.tight_layout()

        plt.show()


        self.roc_auc_plot = (fig, ax)

    def generate_decision_boundary_display(self, X, y):
        self.print("> Generating Decision Boundary Display.")

        estimator = self.split_run.pipeline

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_plot = X.to_numpy()
        else:
            X_plot = np.asarray(X)
            feature_names = ["Feature 1", "Feature 2"]

        y = np.asarray(y)

        if X_plot.shape[1] != 2:
            self.print("\t:: Skipping Decision Boundary (requires exactly 2 features).")
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        DecisionBoundaryDisplay.from_estimator(
            estimator,
            X,
            response_method="predict",
            alpha=0.30,
            ax=ax,
        )

        scatter = ax.scatter(
            X_plot[:, 0],
            X_plot[:, 1],
            c=y,
            edgecolors="k",
            s=40,
        )

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title("Decision Boundary")

        ax.legend(
            *scatter.legend_elements(),
            title="Class",
            loc="best"
        )

        fig.tight_layout()

        self.decision_boundary_plot = (fig, ax)

        plt.show()