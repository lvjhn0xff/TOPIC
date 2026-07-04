
import time
from collections import Counter
import numpy as np 

from utils.python.percentiles import percentiles

from .performance_clf import ClassificationPerformance
from .performance_reg import RegressionPerformance

from utils.python.printing import Printing


class SplitRun(Printing): 
    def __init__(
        self, 

        # Run ID
        id_ = None,

        # Context 
        context = None,

        # Pipeline
        pipeline = None,

        # Splits 
        X_train = None, 
        X_test = None,
        y_train = None, 
        y_test = None,

        # Options 
        plot_decision_boundary = False
    ): 

        # Run ID 
        self.id = id_ 

        # Context
        self.experiment = context 

        # Pipeline 
        self.pipeline = pipeline
        
        # Splits
        self.X_train = X_train 
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test 

        # Evaluation 
        self.performance = None 
        
        # Fit Time 
        self.fit_time = None
        self.predict_train_time = None 
        self.predict_test_time = None 
        self.predict_proba_train_time = None 
        self.predict_proba_test_time = None 

        # Predictions 
        self.predict_train = None 
        self.predict_test = None
        self.predict_proba_train = None 
        self.predict_proba_test = None 

        # Label Distribution 
        self.label_distribution = {}
        self.label_percentiles = {}
        
    def pretraining(self): 
        # Compute number of samples. 
        self.print(f"\tX_train = {self.X_train.shape}")
        self.print(f"\tX_test  = {self.X_test.shape}") 
        self.print(f"\ty_train = {self.y_train.shape}")
        self.print(f"\ty_test  = {self.y_test.shape}")

        # Get label distribution.  
        self.print(f":: Label Distribution")
        if self.experiment.experiment_type == "classification": 
            self.label_distribution = {
                "train" : Counter(self.y_train), 
                "test"  : Counter(self.y_test) 
            }
            
            for split in self.label_distribution: 
                self.print(f"\tSplit [{split}]")
                for class_ in self.label_distribution[split]: 
                    self.print(
                        f"\t\t{class_} = {self.label_distribution[split][class_]}"
                    )

        elif self.experiment.experiment_type == "regression": 
            self.label_distribution = {
                "train" : np.histogram(self.y_train, bins=10)[0], 
                "test"  : np.histogram(self.y_test, bins=10)[0],
                "train_bounds" : np.histogram(self.y_train, bins=10)[1],
                "test_bouhds" : np.histogram(self.y_test, bins=10)[1], 
                "train_percentiles" : \
                    [round(float(x), 2) for x in percentiles(self.y_train) ], 
                "test_percentiles" : \
                    [round(float(x), 2) for x in percentiles(self.y_test)]
            }
            for split in self.label_distribution: 
                self.print(f"\tSplit [{split}] - {self.label_distribution[split]}")
        else: 
            raise Exception(
                f"Unknown experiment type {self.context.experiment_type}."
            )

    def train(self): 
        self.print(f"> Calling .fit() on pipeline.") 
        start = time.time() 
        self.pipeline.fit(self.X_train, self.y_train) 
        end = time.time() 
        self.fit_time = end - start
        self.print(f"> Fitted in {self.fit_time:.5f} seconds.")

    def make_predictions(self):
        self.print(f"> Calling .predict() on X_train")
        start = time.time() 
        self.predict_train = self.pipeline.predict(self.X_train)
        end = time.time() 
        self.predict_train_time = end - start
        self.print(f"> Predicted on X_train in {self.predict_train_time:.4f} seconds.") 

        self.print(f"> Calling .predict() on X_test")
        start = time.time() 
        self.predict_test = self.pipeline.predict(self.X_test) 
        end = time.time() 
        self.predict_test_time = end - start  
        self.print(f"> Predicted on X_test in {self.predict_test_time:.4f} seconds.") 
    
        if hasattr(self.pipeline, "predict_proba"):
            self.print(f"> Calling .predict_proba() on X_train")
            start = time.time() 
            self.predict_proba_train = self.pipeline.predict_proba(self.X_train)
            end = time.time() 
            self.predict_proba_train_time = end - start
            self.print(
                f"> Predicted probabilities on X_train" 
                f" in {self.predict_train_time:.4f} seconds."
            ) 

            self.print(f"> Calling .predict_proba() on X_test")
            start = time.time() 
            self.predict_proba_test = self.pipeline.predict_proba(self.X_test) 
            end = time.time() 
            self.predict_proba_test_time = end - start  
            self.print(
                f"> Predicted probabilities on X_test"
                f" in {self.predict_test_time:.4f} seconds."
            ) 

    def evaluate(self): 
        self.print(f"> Evaluating run.") 

        PerformanceClass = None 
        if self.experiment.experiment_type == "classification": 
            PerformanceClass = ClassificationPerformance 
        elif self.experiment.experiment_type == "regression": 
            PerformanceClass = RegressionPerformance 
        else: 
            raise Exception(
                f"Unknown experiment type: {self.context.experiment_type}"
            )

        self.print(f"> Evaluating on train set.") 
        train_performance = PerformanceClass(
            context = self,
            y = self.y_train, 
            y_hat = self.predict_train,
            y_proba = self.predict_proba_train
        )
        train_performance.indent = self.indent + "\t"
        train_performance.compute()

        self.print(f"> Evaluating on test set.")
        test_performance = PerformanceClass(
            context = self, 
            y = self.y_test, 
            y_hat = self.predict_test,
            y_proba = self.predict_proba_test
        ) 
        test_performance.indent = self.indent + "\t"
        test_performance.compute()