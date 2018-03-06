from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


data_pipeline = Pipeline([
    ("selection", DataFrameSelector(feature_list)),
    ("standardisation", StandardScaler())
])

lin_pipeline = Pipeline([
    ("data_preparation", data_pipeline),
    ("linear", LinearRegression())
])

poly_pipeline = Pipeline([
    ("data_preparation", data_pipeline),
    ("poly", PolynomialFeatures(degree=3)),
    ("linear", LinearRegression())
])


def kfold_train(X, y, k):
    kfold = KFold(n_splits=k)
    
    poly_result = cross_val_score(poly_pipeline, input, output, cv=kfold, scoring="neg_mean_squared_error")
    lin_result = cross_val_score(lin_pipeline, input, output, cv=kfold, scoring="neg_mean_squared_error")

    return lin_result, poly_result
