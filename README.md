<<<<<<< HEAD
# Disaster Response Pipeline
### ETL & NLP Pipeline
In this project, I'm analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages

## 1. Required Libraries <a name="libraries"></a>
Besides the libraries included in the Anaconda distribution for Python 3.6 the following libraries have been included in this project:
* `nltk` 
* `sqlalchemy` 

## 2. Introduction <a name="introduction"></a>
[Figure 8](https://www.figure-eight.com/) helps companies transform they data by providing human annotators and machine learning to annotate data at all scales.
Disaster response is one of events that greatly benefits from data and machine learning modeling. In this project I propose an approach to social media messages annotation.
NLP allows the extraction of great significance in text, understanding how a model classifies and predicts needed responses in disaster cases provides good understanding of the power of words in functional responses.

## 3. Files <a name="files"></a>
Data was downloaded from [Figure 8](https://www.figure-eight.com/dataset/combined-disaster-response-data/).

#### 4. ETL Pipeline <a name="ETL"></a>

File _data/process_data.py_ contains data cleaning pipeline that:

- Loads the `messages` and `categories` dataset
- Merges the two datasets
- Cleans the data
- Stores it in a **SQLite database**

#### 5. ML Pipeline <a name="ML"></a>

File _models/train_classifier.py_ contains machine learning pipeline that:

- Loads data from the **SQLite database**
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a pickle file

#### 6. Flask Web App <a name="Flask"></a>

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python:
data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 

To run ML pipeline that trains classifier and saves python 
models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 

Run the following command in the app's directory to run your web app 
python run.py

Go to http://0.0.0.0:3001/

Notebooks
ETL Pipeline Prep.ipynb - jupyter notebook for data exploration and cleaning
ML Pipeline Preparation - jupyter notebook for model selection and evaluation

## 7. Licensing, Authors, Acknowledgements<a name="licensing"></a>
- Author: Andres Crucetta
- Acknowledgements: Udacity
=======
# disaster_response_pipeline
In this project, I'm anallyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages
>>>>>>> parent of c765ee9... Updated HTML file
