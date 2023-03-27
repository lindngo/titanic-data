## Pandas, Matplotlib, and Seaborn Practice

import os
import io
import warnings

import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

warnings.simplefilter(action='ignore', category=FutureWarning)


titanic = pd.read_csv('https://raw.githubusercontent.com/zariable/data/master/titanic_train.csv')
titanic.head()

"""## Description of the data set
Here's a brief description of each column in the data.

- PassengerID: A column added by Kaggle to identify each row and make submissions easier
- Survived: Whether the passenger survived or not and the value we are predicting (0=No, 1=Yes)
- Pclass: The class of the ticket the passenger purchased (1=1st, 2=2nd, 3=3rd)
- Sex: The passenger's sex
- Age: The passenger's age in years
- SibSp: The number of siblings or spouses the passenger had aboard the Titanic
- Parch: The number of parents or children the passenger had aboard the Titanic
- Ticket: The passenger's ticket number
- Fare: The fare the passenger paid
- Cabin: The passenger's cabin number
- Embarked: The port where the passenger embarked (C=Cherbourg, Q=Queenstown, S=Southampton)
"""

titanic.describe(include='all')

"""### **Question 1: Find the number of missing values for each column.**
The first step in data analysis is to identify columns with missing data. Can you find the columns in this data with missing value as well as the number of records with missing value for each column?  

Hint: you will need [isna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.isna.html) function.
"""

titanic_missing = titanic.isna()
titanic_missing

#Answer: Age (177 records), Cabin (687 records), Embarked (2 records)

#passengerid
display(titanic_missing['PassengerId'].value_counts())

#survived
display(titanic_missing['Survived'].value_counts())

#pclass
display(titanic_missing['Pclass'].value_counts())

#name
display(titanic_missing['Name'].value_counts())

#sex
display(titanic_missing['Sex'].value_counts())

#*age - missing 177 records
display(titanic_missing['Age'].value_counts())

#sibsp
display(titanic_missing['SibSp'].value_counts())

#parch
display(titanic_missing['Parch'].value_counts())

#ticket
display(titanic_missing['Ticket'].value_counts())

#fare
display(titanic_missing['Fare'].value_counts())

#*cabin - missing 687 records
display(titanic_missing['Cabin'].value_counts())

#*embarked - missing 2 records
display(titanic_missing['Embarked'].value_counts())

"""### **Question 2: Impute missing values.**
Now we've identified the following columns with missing values: _Age_, _Cabin_ and _Embarked_. As the next step, we want to impute those missing values. There are three ways to impute the missing values:
- A constant value that has meaning within the domain.
- The mean, median or mode value based on non-missing values of that column.
- A random value drawn from other non-missing values of that column.

Please write code to impute the missing values as follows:
- the missing values of column _age_ with the mean of that column.
- the missing values of column _Cabin_ with a constant value 'other'.
- the missing values of column _Embarked_ with the [mode](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mode.html) of that column.
"""

#missing values of column age with mean of that column
filled_age = titanic['Age'].fillna(titanic['Age'].mean())
filled_age

#missing values of column cabin with constant value 'other'
filled_cabin = titanic['Cabin'].fillna('other')
filled_cabin

#missing values of column embarked with mode of that column
filled_embarked = titanic['Embarked'].fillna(titanic['Embarked'].mode())
filled_embarked

"""### **Question 3: What's the name of the person who has the 16th most expensive ticket?**"""

#Answer:  Farthing, Mr. John

fare_sorted = titanic.sort_values('Fare', ascending = False)
display(fare_sorted)
fare_16 = fare_sorted.iloc[15,:]
display(fare_16)

"""### **Question 4: Out of all the females who survived, what's the name who has the 6th most expensive ticket?**"""

#Answer: Baxter, Mrs. James (Helene DeLaudeniere Chaput)

females = titanic.loc[(titanic['Sex'] == 'female'), :]
females_fare_sorted = females.sort_values('Fare', ascending = False)
female_fare_6 = females_fare_sorted.iloc[5,:]
female_fare_6

"""### **Question 5: Examine the survival rate**
Calculate the survival rate for different gender and Pclass combination and use a couple of sentences to describe your findings. Hint: pivot_table is your friend.
"""

#Answer: Females in Pclass 1 and 2 had a survival rate over 90%, whereas females in Pclass 3 had a 50% survival rate. 
#On the other hand, males had significantly lower survival rates, with Pclass 1 being the highest (36%) and Pclass 2 and 3 being around 14%.

survival_pivot_table = pd.pivot_table(titanic, values = ['Survived'], index = ['Sex', 'Pclass'], margins = True)
survival_pivot_table.drop('All')

"""### **Question 6: Is Age or Fare an important factor to one's chance of survival?**
Visualize the distribution of Column _Age_ for both survived and non-survived population and write down your findings based on the visualization.
"""

#Answer: From this visualization, it is difficult to make significant conclusions based on age. For those who survived, they fell within the 20-40 year old range, where as those who did not survive, they also fell within the 20-40 range.
#On the other hand, we can see a higher density of deceased passengers on the fare chart. This fare chart also has a bigger gap in density between those who survived and are deceased.
sns.displot(titanic, x='Age', hue='Survived', kind = 'kde')
plt.show()

sns.displot(titanic, x='Fare', hue='Survived', kind = 'kde')
plt.show()

"""### **Question 7: Calculate and visualize the survival rate for discrete columns**
- Calculate the survival rate for column _SibSp_ and _Parch_.
- Use sns.barplot to visualize the survival rate for column _SibSp_ and _Parch_.
"""

#Answer: It is interesting to see that the survival rate was highest for those who had 1-3 parents/children, where as 0, 4, 5, and 6 parents/children had lower survival rates.
parch_table = pd.pivot_table(titanic, values = ['Survived'], index = ['Parch'], margins = True)
parch_table = parch_table.drop('All')

sns.barplot(x=parch_table.index, y=parch_table['Survived'])
plt.xlabel('Parch class')
plt.ylabel('Survival Rate')
plt.title('Parch Survival Rate')
plt.show()

#Answer: We can see that 1-2 siblings/spouses had higher survival rates compared to the others.
sibsp_table = pd.pivot_table(titanic, values = ['Survived'], index = ['SibSp'], margins = True)
sibsp_table = sibsp_table.drop('All')

sns.barplot(x=sibsp_table.index, y=sibsp_table['Survived'])
plt.xlabel('SibSp class')
plt.ylabel('Survival Rate')
plt.title('SibSp Survival Rate')
plt.show()

"""### **Question 8: Find the correlations.**
Find the correlations between the feature and the target variable _Survived_ and use heatmap to visualize it. Summarize your findings.
"""

#Answer: From the heatmap generated, we can see that Survived variable is correlated most with Fare, as the correlation value is 0.26. On the other hand, we can conclude that Pclass has the weakest correlation with survival rate with a value of -0.34.

sns.heatmap(titanic.corr(), annot = True)

"""### **Question 9: Any other insights do you draw by analyzing the data? Summarize the findings as well as provide the code leading you to the findings.**"""

#Answer: I was interested to see what the survival rate looked like for the passengers from each of the three ports. From the chart generated below, 
#we can see that Cherbourg had the highest survival rate among males and females compared to the other two ports.

sns.barplot(data = titanic, x = "Embarked", y = "Survived", hue = 'Sex')

"""### **Bonus Point: Build a ML model to predict survival.**
Can you build a logistic regression model to predict the probability of survival for all the passengers in this [file](https://raw.githubusercontent.com/zariable/data/master/titanic_test.csv)? You can evaluate your model accuracy on [Kaggle](https://www.kaggle.com/c/titanic). Can you think of any other ways to improve the model performance? Implement your idea to see if it actually works. 
"""

# TODO