# Goal

# Build a complete AI project:

# Customer Purchase Prediction System


# Project Steps

# Load data
# Clean data
# Train model
# Evaluate accuracy
# Save model
# Predict for new user

# Create a single Python script that:

# Trains a classification model
# Prints accuracy
# Saves the model
# Predicts for a new customer


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# Create dataset

data={
    "customer_visits" : [1,2,3,4,5,6,7,8],
    "purchase": [0,0,0,0,1,1,1,1]
}

df=pd.DataFrame(data)

# Features & target

X=df[["customer_visits"]]
y=df["purchase"]

# Train / Test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Train classification model

model=LogisticRegression()
model.fit(X_train,y_train)

# Evaluate accuracy
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Model Accuracy:" , accuracy*100,"%")

# Save model

joblib.dump(model,"customer_purchase_model.pkl")
print("Model save succuessfully")

# Predict for new customer

new_customer=[[5]]
prediction=model.predict(new_customer)

if prediction==1 :
    print("Prediction: Customer WILL purchase")
else :
    print("Prediction: Customer WILL not purchase")



# Freelance Tip (Important)

# You can describe this project as:

# “Built a customer purchase prediction system using Logistic Regression with model persistence.”

# That sounds professional


