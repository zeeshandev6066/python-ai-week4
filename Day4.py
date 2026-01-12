# Goals

# Understand deployment concept
# Simulate real-world usage
# Prepare for API & web apps


# What is Deployment?

# Deployment means:

# Making your model usable by other people or systems

# Examples:

# Web app
# API
# Command-line tool
# Mobile app backend

# Simple Deployment (Command Line Input)
# Update your script to accept user input:




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

# Take user input
visits = int(input("Enter number of customer visits: "))


new_customer=[[visits]]

prediction=model.predict(new_customer)
probability=model.predict_proba(new_customer)

confidence=probability[0][1]*100

if prediction[0]==1:
    print(f"Customer will purchase {confidence:.2f}% confidence")
else:
    print(f"Customer will not purchase {confidence:.2f}% confidence")