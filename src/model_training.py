from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from data_preprocessing import preprocessed_data 

X, y = preprocessed_data 

y_oVn = y.apply(lambda x: 'Other' if x=='other' else 'Not Other')

X_train, X_test, y_train, y_test = train_test_split(X, y_oVn, test_size=.30, random_state=500)


# Define the parameter grid to search
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'fit_prior': [True, False]
}


# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Increase the number of iterations
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
best_predictions = best_model.predict(X_test)
best_macro_f1 = f1_score(y_test, best_predictions, average='macro')

print(f"Best Parameters: {best_params}")
print(f"Best Macro F1 Score: {best_macro_f1}")


# New Test

dvs_test = X_test
dvs_test['Prediction_1'] = best_predictions

y_dVn = y.apply(lambda x: 'Drunk_Driver' if x=='drunk_driver_involved' else 'Not_Drunk_Driver')

X_train, X_test, y_train, y_test = train_test_split(X, y_dVn, test_size=.30, random_state=500)


# Define the parameter grid to search
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'fit_prior': [True, False]
}


# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Increase the number of iterations
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
best_predictions = best_model.predict(X_test)
best_macro_f1 = f1_score(y_test, best_predictions, average='macro')

print(f"Best Parameters: {best_params}")
print(f"Best Macro F1 Score: {best_macro_f1}")

# New Test

dvs_test['Prediction_2'] = best_predictions

y_sVn = y.apply(lambda x: 'Speeding_Driver' if x=='speeding_driver_involved' else 'Not_Speeding_Driver')

X_train, X_test, y_train, y_test = train_test_split(X, y_sVn, test_size=.30, random_state=500)

# Define the parameter grid to search
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'fit_prior': [True, False]
}


# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Increase the number of iterations
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
best_predictions = best_model.predict(X_test)
best_macro_f1 = f1_score(y_test, best_predictions, average='macro')

print(f"Best Parameters: {best_params}")
print(f"Best Macro F1 Score: {best_macro_f1}")

dvs_test['Prediction_3'] = best_predictions


# Define the final_prediction function
def final_prediction(row):
    if row['Prediction_1'] == 'Other':
        return 'other'
    if row['Prediction_2'] == 'Drunk_Driver':
        return 'drunk_driver_involved'
    if row['Prediction_3'] == 'Speeding_Driver':
        return 'speeding_driver_involved'
    else:
        return 'other'

# Apply the final_prediction function row-wise
dvs_test['Final_Prediction'] = dvs_test.apply(final_prediction, axis=1)

y_test_final = y[y.index.isin(dvs_test.index)]



# Measure Output of MultiNomial Model
cat_lbl = ['drunk_driver_involved', 'speeding_driver_involved','other']
cm = confusion_matrix(y_test_final, dvs_test["Final_Prediction"], labels = cat_lbl )
print(f'\nFinal F1 Score: {f1_score(y_test_final, dvs_test["Final_Prediction"], average="macro", labels=cat_lbl)}')