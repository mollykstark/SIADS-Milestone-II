"""
Python Script to preprocess MET data before training supervised and unsupervised models.
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

numeric_features = ['access_year', 'age']
numeric_transformer = Pipeline(steps=[("imputer",
                                    SimpleImputer(strategy="constant", fill_value=0)),])

categorical_features = ['department','object_name','title','culture','portfolio',
                        'artist_display_name','artist_nationality','mapped_country',
                        'medium', 'artist_gender']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="constant",
                                                                    fill_value="missing")),
                                    ('onehot', OneHotEncoder(handle_unknown="ignore"))])

# tag_features = ['tags']
# tag_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer())])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        # ("tags", tag_transformer, tag_features)
    ]
)

smote = SMOTE(random_state=42)

def supervised_preprocessing(df):
    """
    Function to preprocess data before training a supervised model.

    Args:
        df: A Pandas dataframe of cleaned MET data

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed df data split into 
                                          labels and features, and train and test sets.
        X_train_resampled, y_train_resampled: X_train and y_train after 
                                              undergoing oversampling using SMOTE.
    """
    X = df.drop('is_significant', axis=1)
    y = df['is_significant']

    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled

def unsupervised_preprocessing(df):
    """
    Function to preprocess data before training an unsupervised model.

    Args:
        df: A Pandas dataframe of cleaned MET data

    Returns:
        X: The preprocessed version of df without the label column.
    """
    X = df.drop('is_significant', axis=1)
    X = preprocessor.fit_transform(X)
    return X
