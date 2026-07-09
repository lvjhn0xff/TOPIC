from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
        outer_folds = 10,
        inner_folds = 5,

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
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds

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

        # Repeats 
        self.repeats = [] 

    def describe_dataset(self): 
        self.print(f"\tX = {self.X.shape}")
        self.print(f"\ty = {self.y.shape}")

    def run(self): 
        self.print(f"#" * 80)
        self.print(f"RUNNING EXPERIMENT REPEATS")
        self.print(f"#" * 80)    

        # Create repeat splitter.
        self.print(f"{self.indent}# Creating experiment repeat splitter.")
        self.cv = StratifiedKFold(
            n_splits=self.outer_folds, 
            random_state=self.random_state,
            shuffle=True
        )

        # Loop over splits.
        self.print(f"# Looping over splits.")
        fold_no = 0
    
        self.indent += 1

        for main_index, holdout_index in self.cv.split(self.X, self.y): 
            self.print(f"\t# IN REPEAT {fold_no + 1}")   

            # Split X_train and X_holdout
            self.print(f"\t\t> Splitting dataset to X_train and X_test.")
            X, X_holdout = self.X.iloc[main_index], self.X.iloc[holdout_index] 
            y, y_holdout = self.y.iloc[main_index], self.y.iloc[holdout_index] 

            # Run repetition on main split.
            self.indent += 1 
            self.run_repeat(X, y) 
            self.indent -= 1

            # Increment fold number.
            self.print(f"\t\t> Fold finished.")
            fold_no +=1 


    def run_repeat(self, X, y, repeat_no=1): 
        self.print(f"#" * 80)
        self.print(f"Experiment: {self.experiment_id}")
        self.print(f"#" * 80)

        # Compute and display general statistics about dataset.
        self.print(f"# Describing dataset.")
        self.describe_dataset()

        # Create cross validation splitter.
        self.print(f"# Creating cross validation splitter.")
        self.cv = StratifiedKFold(
            n_splits=self.inner_folds, 
            random_state=self.random_state,
            shuffle=True
        )

        # Loop over splits.
        self.print(f"# Looping over splits.")
        fold_no = 0
    
        for train_index, val_index in self.cv.split(X, y): 
            self.print(f"\t# IN FOLD {fold_no + 1}")   

            # Split X_train and X_test
            self.print(f"\t\t> Splitting dataset to X_train and X_test.")
            X_train, X_val = X.iloc[train_index], X.iloc[val_index] 
            y_train, y_val = y.iloc[train_index], y.iloc[val_index] 

            # Create Pipeline
            self.print(f"\t\t> Creating pipeline.")
            pipeline = self.pipeline_fn(fold_no, X_train, X_val) 

            # Run Split 
            self.print(f"\t\t> Running experiment on split.")
            split_run = SplitRun(
                id_=fold_no, 
                context=self,
                pipeline=pipeline, 
                X_train=X_train,
                X_test=X_val,
                y_train=y_train, 
                y_test=y_val, 
                plot_decision_boundary=X_train.shape[0] == 2
            )
            split_run.set_indent(self.indent + 1)

            # Pre-Training model.
            self.print("> Running pre-training.") 
            split_run.pretraining()

            # Train model.
            self.print("> Training model.") 
            split_run.train()

            # Test model. 
            self.print(f"> Making predictions.")
            split_run.make_predictions()

            # Evaluate model. 
            self.print(f"> Evaluating model") 
            split_run.evaluate()

            # Increment fold number.
            self.print(f"\t\t> Fold finished.")
            fold_no +=1 

      