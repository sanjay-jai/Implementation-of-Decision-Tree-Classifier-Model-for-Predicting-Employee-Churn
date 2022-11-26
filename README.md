# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SANJAI A
RegisterNumber: 212220040142
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
![image](https://user-images.githubusercontent.com/95969295/204103245-455b04d0-ed15-484d-b274-78b45089d210.png)

![image](https://user-images.githubusercontent.com/95969295/204103264-644f421d-e628-4866-b19c-c1a715cdb66e.png)

![image](https://user-images.githubusercontent.com/95969295/204103278-3774aeaf-24ba-4270-b5cf-23fdc3907c2c.png)

![image](https://user-images.githubusercontent.com/95969295/204103297-4fbdc720-fced-4a15-a71b-8a834b18a095.png)

![image](https://user-images.githubusercontent.com/95969295/204103318-930f9250-9370-4eda-8bbc-f0c3331b3844.png)

![image](https://user-images.githubusercontent.com/95969295/204103344-55e3289f-99c2-408e-9d0e-7d02eab7ce34.png)

![image](https://user-images.githubusercontent.com/95969295/204103370-3af4e0e2-8b1d-4d25-8820-f1fc69d732b3.png)

![image](https://user-images.githubusercontent.com/95969295/204103381-95844d99-838d-4864-b573-b93fc858415e.png)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
