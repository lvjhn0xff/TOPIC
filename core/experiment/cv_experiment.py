from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from utils.python.printing import Printing

from .split_run import SplitRun


class CrossValidationExperiment(Printing): 
    def __init__(
        self, 

        # Experiment Id 
        experiment_id = None,

        # Experiment Type 
        experiment_type = "classification",
        
        # Verbose Logging 
        verbose = False,
        indent = "",

        # Data
        X = None, 
        y = None, 

        # Sampling Configuration 
        holdout_size = 0.2, 
        n_splits = 10, 
        n_repeats = 1,

        # Random Seed 
        random_state = 42,

        # Pipeline Generator Function 
        pipeline_fn = None 

    ): 
        # Experiment ID 
        self.experiment_id = experiment_id

        # Experiment Type 
        self.experiment_type = experiment_type

        # Data 
        self.X = pd.DataFrame(X) 
        self.y = pd.Series(y) 

        # Sampling Configuration 
        self.holdout_size = holdout_size
        self.n_splits = n_splits
        self.n_repeats = n_repeats 

        # Cross Validator 
        self.cv = None

        # Random Seed 
        self.random_state = random_state

        # Pipeline Generator Function 
        self.pipeline_fn = pipeline_fn

        # List of Classes 
        self.classes = np.unique(self.y)
        self.class_count = len(list(set(self.y)))

        # Whether multiclass or binary
        self.multiclass = self.class_count > 2
        self.binary = self.class_count == 2

    def describe_dataset(self): 
        self.print(f"\tX = {self.X.shape}")
        self.print(f"\ty = {self.y.shape}")

    def run(self): 
        self.print(f"#" * 80)
        self.print(f"Experiment: {self.experiment_id}")
        self.print(f"#" * 80)

        # Compute and display general statistics about dataset.
        self.print(f"{self.indent}# Describing dataset.")
        self.describe_dataset()

        # Create cross validation splitter.
        self.print(f"{self.indent}# Creating cross validation splitter.")
        self.cv = RepeatedStratifiedKFold(
            n_splits=self.n_splits, 
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        # Loop over splits.
        self.print(f"# Looping over splits.")
        fold_no = 0
        for train_index, test_index in self.cv.split(self.X, self.y): 
            self.print(f"\t# IN FOLD {fold_no + 1}")   

            # Split X_train and X_test
            self.print(f"\t\t> Splitting dataset to X_train and X_test.")
            X_train, X_test = self.X.loc[train_index], self.X.loc[test_index] 
            y_train, y_test = self.y.loc[train_index], self.y.loc[test_index] 

            # Create Pipeline
            self.print(f"\t\t> Creating pipeline.")
            pipeline = self.pipeline_fn(fold_no, X_train, X_test) 

            # Run Split 
            self.print(f"\t\t> Running experiment on split.")
            split_run = SplitRun(
                id_=fold_no, 
                context=self,
                pipeline=pipeline, 
                X_train=X_train,
                X_test=X_test,
                y_train=y_train, 
                y_test=y_test, 
                plot_decision_boundary=X_train.shape[0] == 2
            )
            split_run.set_indent(self.indent + "\t\t")

            # Pre-Training model.
            print(f"{self.indent}\t\t> Running pre-training.") 
            split_run.pretraining()

            # Train model.
            print(f"{self.indent}\t\t> Training model.") 
            split_run.train()

            # Test model. 
            print(f"{self.indent}\t\t> Making predictions.")
            split_run.make_predictions()

            # Evaluate model. 
            print(f"{self.indent}\t\t> Evaluating model") 
            split_run.evaluate()

            # Increment fold number.
            self.print(f"\t\t> Fold finished.")
            fold_no +=1 



