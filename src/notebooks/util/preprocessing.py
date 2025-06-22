"""
Python Script to preprocess MET data before training supervised and unsupervised models.
"""
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

categorical_features = ['department','object_name','title','culture','portfolio',
                        'artist_display_name','artist_nationality','country_mapped',
                        'medium', 'artist_gender']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="constant",
                                                                    fill_value="missing")),
                                    ('onehot', OneHotEncoder(handle_unknown="ignore"))])

# tag_features = ['tags']
# tag_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer())])

supervised_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        # ("tags", tag_transformer, tag_features)
    ]
)

unsupervised_preprocessor = ColumnTransformer(
    transformers=[
        ("num", scaled_numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
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

    X = supervised_preprocessor.fit_transform(X)

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
    X = unsupervised_preprocessor.fit_transform(X)
    return X

def supervised_feature_importance(clf):
    """
    Function to get feature importance from a trained classifer.

    Args:
        clf: A trained classifier

    Returns:

    """
    cat_ohe = supervised_preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_feature_names = cat_ohe.get_feature_names_out(categorical_features)

    grouped_feature_indices = defaultdict(list)

    for idx, feat in enumerate(ohe_feature_names):
        orig_col = feat.split('_')[0]
        grouped_feature_indices[orig_col].append(idx)
    # all_feature_names = list(categorical_feature_names) + numeric_features

    # importances = clf.feature_importances_
    ohe_importances = clf.feature_importances_[:len(ohe_feature_names)]

    grouped_importances = {
        col: sum(ohe_importances[i] for i in indices)
        for col, indices in grouped_feature_indices.items()
    }

    numeric_importances = clf.feature_importances_[len(ohe_feature_names):]

    for col, importance in zip(numeric_features, numeric_importances):
        grouped_importances[col] = importance

    grouped_df = pd.DataFrame({
        'feature': list(grouped_importances.keys()),
        'importance': list(grouped_importances.values())
    }).sort_values(by='importance', ascending=False)

    return grouped_df

def unsupervised_feature_importance(pca):
    """
    Function to get feature importance from a trained classifer.

    Args:
        pca: A fitted PCA object

    Returns:

    """
    ohe = unsupervised_preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe.get_feature_names_out(categorical_features)

    # Combine with numeric features
    all_feature_names = np.concatenate([numeric_features, ohe_feature_names])

    pca_importance = np.sum(
        np.abs(pca.components_.T) * pca.explained_variance_ratio_, axis=1
    )

    aggregated_importance = defaultdict(float)

    for name, importance in zip(all_feature_names, pca_importance):
        if name in numeric_features:
            aggregated_importance[name] += importance
        else:
            # Example: 'gender_female' → 'gender'
            for cat_col in categorical_features:
                if name.startswith(cat_col + '_'):
                    aggregated_importance[cat_col] += importance
                    break
       
    importance_df = pd.DataFrame(
        list(aggregated_importance.items()),
        columns=["feature", "importance"]
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df
