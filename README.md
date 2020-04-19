
# Disaster Response Pipeline Project

  

### Motivation

This project was made for the **Data Science Nanodegree Program**, and works for categorize messages during a disaster situation.

  

### Dependencies

- Python

- NumPy

- Pandas

- Sciki-Learn

- NLTK

- SQLalchemy

- Flask

- Plotly

- Bootstrap

  

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

  

- To run ETL pipeline that cleans data and stores in database

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

- To run ML pipeline that trains classifier and saves

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  

2. Run the following command in the app's directory to run your web app.

`python run.py`

  

3. Go to http://0.0.0.0:3001/

 
  

### Licensing, Authors, Acknowledgements

This project uses [Figure Eight](https://appen.com/)  datasets.
This project was made by: [Matheus Morett](https://github.com/matheusmorett) using the [Udacity](https://www.udacity.com/) knowledge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Feel free to contribute =].