from core.experiment.cv_experiment import CrossValidationExperiment 
from sklearn.preprocessing import StandardScaler

from core.datasets.clf_2d import load_clf_2d
from core.datasets.reg_2d import load_reg_2d
from core.datasets.openml import load_openml

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

experiment_type = "classification"

X, y, make_preprocessor, info = load_openml("telco-customer-churn")
# X, y = load_clf_2d("circles") 
# X, y = load_reg_2d("moons")

def make_pipeline(fold_no, X_train, X_test):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("balancer", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(hidden_layer_sizes=(100,), verbose=True, random_state=42))
    ])

experiment = CrossValidationExperiment(
    experiment_id = "Unnamed Experiment", 
    X = X, 
    y = y, 
    experiment_type=experiment_type,
    pipeline_fn = make_pipeline
)

experiment.run()