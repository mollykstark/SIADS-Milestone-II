"""
Python Script to preprocess MET data before training supervised and unsupervised models.
"""
# pylint: disable=C0103
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

pd.options.display.float_format = '{:.3f}'.format

numeric_features = ['access_year', 'age']
numeric_transformer = Pipeline(steps=[("imputer",
                                    SimpleImputer(strategy="constant", fill_value=0)),])

scaled_numeric_transformer = Pipeline(steps=[("imputer",
                                    SimpleImputer(strategy="constant", fill_value=0)),
                                    ('scaler', StandardScaler())])

categorical_features = ['department','object_name','title','portfolio',
                        'artist_display_name','country_mapped',
                        'medium', 'artist_gender','culture','artist_nationality',]
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="constant",
                                                                    fill_value="missing")),
                                    ('onehot', OneHotEncoder(handle_unknown="ignore"))])

# tag_features = ['tags']
# tag_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer())])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", scaled_numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        # ("tags", tag_transformer, tag_features)
    ]
)

smote = SMOTE(random_state=42)

def supervised_preprocessing(df, percent):
    """
    Function to preprocess data before training a supervised model.

    Args:
        df: A Pandas dataframe of cleaned MET data

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed df data split into 
                                          labels and features, and train and test sets.
        X_train_resampled, y_train_resampled: X_train and y_train after 
                                              undergoing oversampling using SMOTE.
        X, y: Feature set after preprocessing and label set before being broken up.
    """
    X = df.drop('is_significant', axis=1)
    y = df['is_significant']

    X = preprocessor.fit_transform(X)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=percent,random_state=42,stratify=y)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled, X, y

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

def supervised_feature_importance(clf):
    """
    Function to get feature importance from a trained supervised classifer.

    Args:
        clf: A trained classifier

    Returns:
        grouped_df: A dataframe with all features and their importances.
    """
    cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_feature_names = cat_ohe.get_feature_names_out(categorical_features)

    grouped_feature_indices = defaultdict(list)
    for idx, feat in enumerate(ohe_feature_names):
        for col in categorical_features:
            if feat.startswith(col + '_'):
                grouped_feature_indices[col].append(idx)
                break

    coeffs = np.abs(clf.coef_[0])
    ohe_coeffs = coeffs[:len(ohe_feature_names)]
    numeric_coeffs = coeffs[len(ohe_feature_names):]

    grouped_importances = {
        col: sum(ohe_coeffs[i] for i in indices)
        for col, indices in grouped_feature_indices.items()
    }

    for col, importance in zip(numeric_features, numeric_coeffs):
        grouped_importances[col] = importance

    grouped_df = pd.DataFrame({
        'feature': list(grouped_importances.keys()),
        'importance': list(grouped_importances.values())
    }).sort_values(by='importance', ascending=False)

    grouped_df['importance'] = grouped_df['importance'] / grouped_df['importance'].sum()

    return grouped_df

def unsupervised_feature_importance(pca):
    """
    Function to get feature importance from PCA.

    Args:
        pca: A fitted PCA object

    Returns:
        importance_df: A dataframe with all features and their importances.
    """
    ohe = preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numeric_features, ohe_feature_names])

    pca_importance = np.sum(np.abs(pca.components_.T) * pca.explained_variance_ratio_, axis=1)
    aggregated_importance = defaultdict(float)

    for name, importance in zip(all_feature_names, pca_importance):
        if name in numeric_features:
            aggregated_importance[name] += importance
        else:
            for cat_col in categorical_features:
                if name.startswith(cat_col + '_'):
                    aggregated_importance[cat_col] += importance
                    break

    importance_df = pd.DataFrame(
        list(aggregated_importance.items()),
        columns=["feature", "importance"]
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df

def ablation_testing(df, new_categorical_features, percent):
    """
        Function to do ablation testing.

    Args:
        df: Pandas dataframe
        new_categorical_features: A list of to use in the model
        percent: training percent

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed df data split into 
                                          labels and features, and train and test sets.
        X_train_resampled, y_train_resampled: X_train and y_train after 
                                              undergoing oversampling using SMOTE.
        X, y: Feature set after preprocessing and label set before being broken up.
    """
    preprocessor_ab = ColumnTransformer(
        transformers=[
            ("num", scaled_numeric_transformer, numeric_features),
            ("cat", categorical_transformer, new_categorical_features),
        ]
    )
    X = df.drop('is_significant', axis=1)
    y = df['is_significant']

    X = preprocessor_ab.fit_transform(X)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=percent,random_state=42,stratify=y)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled, X, y
