import sys

# import libraries
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

# downloads
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
  

def load_data(database_filepath):
    """
    Description: 
        Load Data
    
    Input:
        database_filepath: database filepath to connect
        
    Output:
        X: feature DataFrame
        Y: label DataFrame
        category_names: category names to show in the application
    """
    # enter into the database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # read the right table
    df =  pd.read_sql_table('messages', engine)
    
    # set X and Y values 
    X = df.message.values
    Y = df.iloc[:,5:]
    
    # set label names
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """
    Description: 
        Tranform the string in tokens
    
    Input:
        text: text to be tokenized
    
    Output:
        clean_tokens: clean tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = nltk.word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = nltk.WordNetLemmatizer() 

    # iterate through each token
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]    

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def build_model():
    """
    Description:
        Create the ML pipeline model
     
    Input:
        None
        
    Outpt:
        GridSearchCV: Grid search model object
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Create grid search parameters
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', n_jobs=-1)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Description:
        Evaluate the model value
    
    Input:
        model
        X_test
        y_test
        category_names
    
    Output:
        None
    """
    # Predict the model, and print.
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Description:
        Save model as a pickle file
    
    Input:
        model
        model_filepath
    
    Output:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    print(model_filepath)


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
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()