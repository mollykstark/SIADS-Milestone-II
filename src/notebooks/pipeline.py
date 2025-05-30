"""
Python Script to create a column labeling the country of origin for MET artworks.
"""
from met_cleaning import clean_met_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

df = clean_met_data()

X = df.drop('is_significant', axis=1)
y = df['is_significant']

numeric_features = ['accession_year', 'extracted_date']
numeric_transformer = Pipeline(steps=[("imputer",
                                       SimpleImputer(strategy="constant", fill_value=0)),])

categorical_features = ['department','object_name','title','culture','portfolio',
                        'artist_display_name','artist_nationality','mapped_country',
                        'medium', 'artist_gender']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="constant",
                                                                    fill_value="missing")),
                                      ('onehot', OneHotEncoder(handle_unknown="ignore"))])

tag_features = ['tags']
tag_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer())])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        # ("tags", tag_transformer, tag_features)
    ]
)

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', BalancedRandomForestClassifier())
                       ])

smote = SMOTE(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

pipe.fit(X_train_resampled, y_train_resampled)

metrics = {
    "train_data":{
        "score": pipe.score(X_train, y_train),
        "mae": mean_absolute_error(y_train, pipe.predict(X_train))
    },
    "test_data":{
        "score": pipe.score(X_test, y_test),
        "mae": mean_absolute_error(y_test, pipe.predict(X_test))
    }
}

print(metrics)
