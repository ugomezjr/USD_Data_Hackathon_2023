# USD Data Hackathon 2023
------
## Authors:  Ned Kost, Ulises Gomez
------
## Project Description
------
This code was created as an entry for a Virtual Data Hackathon held by the University of San Diego.  The goal of the hackathon was to develop a robust model predicting the primary factor influencing fata car crashes: drunk driving, speeding, or other factors.  We were provided 2 days to create, test, and submit our entries.  

We used an XGBoost Classifier model to predict the influencing factor based on the Fatality Analysis Reporting System(FARS) data provided for the challenge.  

------
Here are the specifics of this implementation:
- Data was loaded from the provided .csv file
- We used hot encoding to change categorical data into binary data representations
- An XGBoost Multiclass Classifier algorithm was built from the xgboost package.
- Since this classification is imbalanced, with most influencing factors being "Other", we created sample weights to better tune the classifier.
- Model accuracy was calculated based on the macro F1 score 
------
#Project Structure
- `data/`:  This directory contains data provided to us for Training and Testing our model.
- `docs/`:  This directory contains the schema documentation for the FARS dataset
- `notebooks/`: This direcotry contains the Jupiter notebooks we used to build and test our model.
------
#License
------
This project is privded under the Apache License 2.0.  Please see the [LICENSE](https://github.com/NedKost/usd_data_hackathon_2023/blob/main/LICENSE) file for more information.
