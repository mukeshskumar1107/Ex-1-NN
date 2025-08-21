<H3>NAME: MUKESH KUMAR S</H3>
<H3>REGISTER NO: 212223240099</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 21.08.2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv("Churn_Modelling.csv")

# Checking Data
df.head()
df.tail()
df.columns


# Finding Missing 
df.isnull().sum()

#Handling Missing values
df.fillna(method='ffill',inplace=True)

#Check for Duplicates
df.duplicated()

#Detect Outliers
df.describe()

#Normalize the dataset
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

#split the dataset into input and output
X=df1.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)

#splitting the data for training & Testing
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#Print the training data and testing data
print("X_train")
print(x_train)
print("Length of X_train",len(x_train))
print("X_test")
print(x_test)
print("Length of X_test",len(x_test))
```
## OUTPUT:

 Data checking:
 
<img width="509" height="71" alt="image" src="https://github.com/user-attachments/assets/202de1e5-4b74-4eeb-be79-275ecfe9fd6a" />

Missing Data:

<img width="163" height="218" alt="image" src="https://github.com/user-attachments/assets/cedf6c9c-da28-4767-9487-dee1d212ea83" />

Duplicates identification:

<img width="179" height="180" alt="image" src="https://github.com/user-attachments/assets/ab1a620b-b7b9-45ee-a9ac-13b486e37643" />

Vakues of 'Y':

<img width="196" height="33" alt="image" src="https://github.com/user-attachments/assets/b3053c85-743a-486d-a9ac-5130ed31bb95" />

Outliers:

<img width="1036" height="243" alt="image" src="https://github.com/user-attachments/assets/6d6e54b8-a17a-4386-80de-309d6ef242c9" />

Checking datasets after dropping string values data from dataset:

<img width="906" height="174" alt="image" src="https://github.com/user-attachments/assets/aceca48e-1216-4549-80d9-34868c9dd32c" />

Normalize the dataset:

<img width="513" height="385" alt="image" src="https://github.com/user-attachments/assets/e94733bb-5bb5-4ac6-a233-ecea7773b0c1" />

Split the dataset:

<img width="301" height="128" alt="image" src="https://github.com/user-attachments/assets/c973bd0e-8eb7-400e-a0e6-7334b09c2d02" />

Training and testing model:

<img width="343" height="339" alt="image" src="https://github.com/user-attachments/assets/f751ae87-0f89-4c1e-bd7f-14154804d033" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


