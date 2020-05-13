import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
import pickle

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.corpus import stopwords


def load_data(database_filepath):
    """
    Load the data

    Inputs:
    database_filepath: String. Filepath for the db file containing the cleaned data.

    Output:
    X: dataframe. Contains the feature data.
    y: dataframe. Contains the labels (categories) data.
    category_names: List of strings. Contains the labels names.
    """

    df = pd.read_sql_table(database_filepath, engine)
    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    y.drop(columns=["related", "other_infrastructure", "other_weather", "other_aid", "direct_report", "weather_related"])
    category_names = y.columns.tolist()
    return X, y, category_names

def tokenize(text):
    """
    Normalize, tokenize and stems texts.

    Input:
    text: string. Sentence containing a message.

    Output:
    stemmed_tokens: list of strings. A list of strings containing normalized and stemmed tokens.
    """

    # creating stop words
    stop_words = set(stopwords.words("english"))

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


class StartVerbExtractor(BaseEstimator, TransformerMixin):
    def start_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) != 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tag = pd.Series(X).apply(self.start_verb)
        return pd.DataFrame(X_tag)

def build_model():
    """
    Builds a ML pipeline and performs gridsearch.
    Args:
    None
    Returns:
    cv: gridsearchcv object.
    """
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('starting_verb', StartVerbExtractor())
            ])),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
        ])
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    """
    Returns test accuracy, number of 1s and 0s, recall, precision and F1 Score.

    Inputs:
    model: model object. Instanciated model.
    X_test: pandas dataframe containing test features.
    y_test: pandas dataframe containing test labels.
    category_names: list of strings containing category names.

    Returns:
    None
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()