# Goals

# You will learn to:

# Show prediction probability
# Explain confidence to clients
# Make outputs more professional


# Why Probability Matters

# Instead of just:

# “Customer WILL purchase”
# You can say:

# “Customer will purchase with 82% confidence”
# Clients love this.

# Update Your Script
# Add prediction:


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

new_customer=[[6]]
prediction=model.predict(new_customer)

if prediction==1 :
    print("Prediction: Customer WILL purchase")
else :
    print("Prediction: Customer WILL not purchase")


# Probability prediction

probability=model.predict_proba(new_customer)
confidence=probability[0][1]*100
print(f"Purchase Probability : {confidence:.2f}%")
