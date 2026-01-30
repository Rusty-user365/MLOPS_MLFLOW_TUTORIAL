import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow




#load wine dataset

wine=load_wine()

X=wine.data
y=wine.target

#Train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=42)


#define the params for RF Model

max_depth=5
n_estimators=5




with mlflow.start_run():
    #Create RF Model
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    
    #Fit the model
    rf.fit(X_train,y_train)
    
    #Predict the model
    y_pred=rf.predict(X_test)
    
    #Calculate accuracy
    acc=accuracy_score(y_test,y_pred)
    
    #Log parameters and metrics
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_metric("accuracy",acc)
    # mention your experiment below
   


    # creating a confusuion matrix plot  ctrl+/
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')   
    plt.savefig('confusion_matrix.png')
    # log artifacts using mlflow

    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)


    # tags
    mlflow.set_tag("Author", "Abhishek")
    
    mlflow.set_tag("Model", "RandomForestClassifier")
    mlflow.set_tag("Dataset", "Wine Dataset")

    # log the model
    mlflow.sklearn.log_model(rf,"random_forest_wine_model")

    print(acc)
