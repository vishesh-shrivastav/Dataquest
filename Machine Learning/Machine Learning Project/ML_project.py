import pandas as pd
loans_2007 = pd.read_csv("loans_2007.csv")
loans_2007.drop_duplicates()
print(loans_2007.head(1))
print(loans_2007.shape[1])

##################################################### Part 1 - Data Cleaning ######################################################
# Remove irrelevant columns
loans_2007 = loans_2007.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
loans_2007 = loans_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)
loans_2007 = loans_2007.drop(["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", 
							  "last_pymnt_d", "last_pymnt_amnt"], axis=1)

# Display first row and number of columns
print(loans_2007.head(1))
print(loans_2007.shape[1])

# Target column - loan_status
print(loans_2007['loan_status'].value_counts())

# Remove loans other than "Fully paid" and "Charged Off"
loans_2007 = loans_2007[(loans_2007['loan_status'] == "Fully Paid") | (loans_2007['loan_status'] == "Charged Off")]

# Encode "Fully paid" and "Charged Off" to 1 and 0
status_replace = {
    "loan_status" : {
        "Fully Paid": 1,
        "Charged Off": 0,
    }
}

loans_2007 = loans_2007.replace(status_replace)

# Remove columns that contain only one unique value
orig_columns = loans_2007.columns
drop_columns = []
for col in orig_columns:
    col_series = loans_2007[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
loans_2007 = loans_2007.drop(drop_columns, axis=1)
print(drop_columns)

# To CSV
loans_2007.to_csv('filtered_loans_2007.csv')

################################################## Part 2 - Preparing the Features ################################################
loans = pd.read_csv('filtered_loans_2007.csv')

# Find number of missing values across the data
null_counts = loans.isnull().sum()
print(null_counts)

# Remove "pub_rec_bankruptcies" since it has 697 missing values
loans = loans.drop("pub_rec_bankruptcies", axis=1)
# Remove all rows from data that have any missing value
loans = loans.dropna(axis=0)
print(loans.dtypes.value_counts())

# Select object type columns as they contain text values
object_columns_df = loans.select_dtypes(include=["object"])
print(object_columns_df.head(0))

# Explore frequency of values in these columns
cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']
for c in cols:
    print(loans[c].value_counts())

# Look at the unique value counts for the purpose and title columns to understand which column we want to keep
print(loans["purpose"].value_counts())
print(loans["title"].value_counts())

# Remove more irrelevant columns
loans = loans.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)

# Convert "int_rate" and "revol_util" to float
loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")

# Encode values in "emp_length" column
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
loans = loans.replace(mapping_dict)

# Carry out one hot encoding for "home_ownership", "verification_status", "purpose", "term"
cat_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(loans[cat_columns])
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(cat_columns, axis=1)
loans.to_csv("cleaned_loans_2007.csv")

################################################### Part 3 - Making Predictions ###################################################
loans = pd.read_csv("cleaned_loans_2007.csv")
print(loans.info())

# Some predictions are stored in a NumPy array called predictions
# Extract FP, TP, FN, TN

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Compute FP and TP to verify
import numpy

# Predict that all loans will be paid off on time.
predictions = pd.Series(numpy.ones(loans.shape[0]))

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr) #1.0
print(fpr) #1.0

# Fit logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

cols = loans.columns
train_cols = cols.drop("loan_status")

# Dataframe that has all feature columns
features = loans[train_cols]
# Dataframe with target column
target = loans["loan_status"]
# Fir lr model
lr.fit(features, target)
# Make predictions
predictions = lr.predict(features)

# This model has overfitting since we are using the same data to fit model and make predictions
# Improve model by using k-fold cross validation
lr = LogisticRegression()
predictions = cross_val_predict(lr, features, target, cv=3)
predictions = pd.Series(predictions)
# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])
# Rates
tpr = tp  / (tp + fn)
fpr = fp  / (fp + tn)
print(tpr) # 0.9989121566494424
print(fpr) # 0.9967943009795192

# Handling imbalance of classes
# Penalize misclassification of minority class (loan_status = 0) during here
# The penalty means that the classifier pays more attention to correctly classifying rows where loan_status is 0
# This lowers accuracy when loan_status is 1, but raises accuracy when loan_status is 0

lr = LogisticRegression(class_weight="balanced")
predictions = cross_val_predict(lr, features, target, cv=3)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr) # 0.6636146617109359
print(fpr) # 0.38664292074799644

# Lower fpr by assigning harsher penalty by manually setting penalty values
penalty = {
    0: 10,
    1: 1
}

lr = LogisticRegression(class_weight=penalty)
predictions = cross_val_predict(lr, features, target, cv=3)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr) # 0.24267972078687336
print(fpr) # 0.09154051647373107

# Try Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight = "balanced", random_state=1)
predictions = cross_val_predict(rf, features, target, cv=3)
predictions = pd.Series(predictions)

# False positives
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr) # 0.9708699725017376
print(fpr) # 0.9271593944790739