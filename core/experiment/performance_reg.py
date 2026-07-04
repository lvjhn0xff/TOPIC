from sklearn.metrics import *
import numpy as np
from utils.python.percentiles import percentiles
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
import pandas as pd

from utils.python.printing import Printing

class RegressionPerformance(Printing): 
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
        self.scores = {}
    
        # Dataset 
        self.y = y 
        self.y_hat = y_hat 
        self.y_proba = y_proba
        self.residuals = np.abs(self.y_hat - self.y)

    def compute(self): 
        self.print(f"> Computing performance.")
        self.compute_rmse_score() 
        self.compute_r2_score() 
        self.compute_mae_score() 
        self.compute_mape_score() 
        self.compute_mse_score() 
        self.compute_rmsle_score() 
        self.compute_adjusted_r2_score() 
        self.compute_residuals() 
        self.generate_error_plot()

        self.generate_decision_boundary_display(
            self.split_run.X_train, self.split_run.y_train
        ) 
        self.generate_decision_boundary_display(
            self.split_run.X_test, self.split_run.y_test
        ) 
    
    def compute_rmse_score(self): 
        self.print(f"> Computing RMSE Score")
        score = root_mean_squared_error(self.y, self.y_hat)
        self.scores["rmse"] = score
        self.print(f"\t:: Result: {score}")
    
    def compute_r2_score(self): 
        self.print(f"> Computing R2 Score.")
        score = r2_score(self.y, self.y_hat)
        self.scores["r2"] = score
        self.print(f"\t:: Result: {score}")
        
    def compute_mae_score(self): 
        self.print(f"> Computing MAE Score.") 
        score = mean_absolute_error(self.y, self.y_hat)
        self.scores["mae"] = score
        self.print(f"\t:: Result: {score}")

    def compute_mape_score(self):
        self.print(f"> Computing MAPE Score.")
        score = mean_absolute_percentage_error(self.y, self.y_hat)
        self.scores["mae"] = score
        self.print(f"\t:: Result: {score}")

    def compute_mse_score(self): 
        self.print(f"> Computing MSE Score.")
        score = mean_squared_error(self.y, self.y_hat)
        self.scores["mae"] = score
        self.print(f"\t:: Result: {score}")

    def compute_rmsle_score(self): 
        try:
            self.print(f"> Computing RMSLE Score.")
            score = root_mean_squared_log_error(self.y, self.y_hat)
            self.scores["mae"] = score
            self.print(f"\t:: Result: {score}")
        except: 
            self.print(f"> RMSLE failed or not applicable, skipping...")

    def compute_adjusted_r2_score(self):
        self.print(f"> Computing Adjusted R2 Score.")

        n = len(self.y)
        p = self.split_run.experiment.class_count

        if p is None or n <= p + 1:
            score = float("nan")
        else:
            r2 = r2_score(self.y, self.y_hat)
            score = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        self.scores["adjusted_r2"] = score
        self.print(f"\t:: Result: {score}")

    def compute_residuals(self):
        self.print(f"> Computing Residuals.")
        hist = np.histogram(self.residuals, bins=10)
        self.residuals = {
            "histogram" : [round(x, 2) for x in hist[0]],
            "histogram_bounds" : [round(x, 2) for x in hist[1]], 
            "percentiles" : percentiles(self.residuals)
        }
        self.print(f"\t :: Histogram   : {self.residuals['histogram']}")
        self.print(f"\t :: Percentiles : {self.residuals['percentiles']}")

    def generate_error_plot(self):
        self.print("> Generating Error Plot")

        residuals = self.y_hat - self.y

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # ------------------------------------------------------------------
        # True vs Predicted
        # ------------------------------------------------------------------
        axs[0].scatter(self.y, self.y_hat, alpha=0.7)

        mn = min(np.min(self.y), np.min(self.y_hat))
        mx = max(np.max(self.y), np.max(self.y_hat))

        axs[0].plot([mn, mx], [mn, mx], "--")
        axs[0].set_title("True vs Predicted")
        axs[0].set_xlabel("True")
        axs[0].set_ylabel("Predicted")

        # ------------------------------------------------------------------
        # Residual Plot
        # ------------------------------------------------------------------
        axs[1].scatter(self.y_hat, residuals, alpha=0.7)
        axs[1].axhline(0.0, linestyle="--")
        axs[1].set_title("Residual Plot")
        axs[1].set_xlabel("Predicted")
        axs[1].set_ylabel("Residual")

        # ------------------------------------------------------------------
        # Residual Histogram
        # ------------------------------------------------------------------
        axs[2].hist(residuals, bins=30)
        axs[2].set_title("Residual Distribution")
        axs[2].set_xlabel("Residual")
        axs[2].set_ylabel("Frequency")

        plt.tight_layout()

        # Save figure and axes
        self.error_plot = (fig, axs)

        plt.show()

        self.print("\t:: Error plot generated.")    

    def generate_decision_boundary_display(self, X, y):
        self.print("> Generating Prediction Surface.")

        estimator = self.split_run.pipeline

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_plot = X.to_numpy()
        else:
            X_plot = np.asarray(X)
            feature_names = ["Feature 1", "Feature 2"]

        y = np.asarray(y)

        if X_plot.shape[1] != 2:
            self.print("\t:: Skipping Prediction Surface (requires exactly 2 features).")
            return

        x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
        y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300),
        )

        grid = np.c_[xx.ravel(), yy.ravel()]

        if isinstance(X, pd.DataFrame):
            grid = pd.DataFrame(grid, columns=X.columns)

        zz = estimator.predict(grid)
        zz = np.asarray(zz).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 6))

        contour = ax.contourf(
            xx,
            yy,
            zz,
            levels=30,
            alpha=0.6,
        )

        scatter = ax.scatter(
            X_plot[:, 0],
            X_plot[:, 1],
            c=y,
            edgecolors="k",
            s=40,
            norm=Normalize(vmin=y.min(), vmax=y.max()),
        )

        fig.colorbar(contour, ax=ax, label="Predicted Value")
        fig.colorbar(scatter, ax=ax, label="True Value")

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title("Regression Prediction Surface")

        fig.tight_layout()

        self.decision_boundary_plot = (fig, ax)

        plt.show()