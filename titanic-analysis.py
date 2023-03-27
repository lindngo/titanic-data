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

""" Understanding dataset

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

# Identifying columns with missing data 

titanic_missing = titanic.isna()
titanic_missing

# Imputing missing values using different methods: 
# using a meaningful constant value to the dataset, 
# using mean, median, or mode based on non-missing values of respective column, or
# using random value based on non-missing values of the respective column.

# Column cabin with meaingful constant value 'other'
filled_cabin = titanic['Cabin'].fillna('other')
filled_cabin

# Column age with mean of that column
filled_age = titanic['Age'].fillna(titanic['Age'].mean())
filled_age

# Column embarked with mode of that column
filled_embarked = titanic['Embarked'].fillna(titanic['Embarked'].mode())
filled_embarked

# Practice

# Q: What's the name of the person who has the 16th most expensive ticket?

fare_sorted = titanic.sort_values('Fare', ascending = False)
fare_16 = fare_sorted.iloc[15,:]
fare_16

# Q: Out of all the females who survived, what's the name who has the 6th most expensive ticket?

females = titanic.loc[(titanic['Sex'] == 'female'), :]
females_fare_sorted = females.sort_values('Fare', ascending = False)
female_fare_6 = females_fare_sorted.iloc[5,:]
female_fare_6

# Q: Examine the survival rate. Calculate the survival rate for different gender and Pclass combination.

survival_pivot_table = pd.pivot_table(titanic, values = ['Survived'], index = ['Sex', 'Pclass'], margins = True)
survival_pivot_table.drop('All')

# A: Females in Pclass 1 and 2 had a survival rate over 90%, whereas females in Pclass 3 had a 50% survival rate. 
# On the other hand, males had significantly lower survival rates, with Pclass 1 being the highest (36%) and Pclass 2 and 3 being around 14%.

# Q: Is Age or Fare an important factor to one's chance of survival?

sns.displot(titanic, x='Age', hue='Survived', kind = 'kde')
plt.show()

sns.displot(titanic, x='Fare', hue='Survived', kind = 'kde')
plt.show()

# A: From this visualization, it is difficult to make significant conclusions based on age. For those who survived, they fell within the 20-40 year old range, where as those who did not survive, they also fell within the 20-40 range.
# On the other hand, we can see a higher density of deceased passengers on the fare chart. This fare chart also has a bigger gap in density between those who survived and are deceased.

# Q: Calculate and visualize the survival rate for discrete columns. Calculate the survival rate for column _SibSp_ and _Parch_.

parch_table = pd.pivot_table(titanic, values = ['Survived'], index = ['Parch'], margins = True)
parch_table = parch_table.drop('All')

sns.barplot(x=parch_table.index, y=parch_table['Survived'])
plt.xlabel('Parch class')
plt.ylabel('Survival Rate')
plt.title('Parch Survival Rate')
plt.show()


# A: It is interesting to see that the survival rate was highest for those who had 1-3 parents/children, where as 0, 4, 5, and 6 parents/children had lower survival rates.

sibsp_table = pd.pivot_table(titanic, values = ['Survived'], index = ['SibSp'], margins = True)
sibsp_table = sibsp_table.drop('All')

sns.barplot(x=sibsp_table.index, y=sibsp_table['Survived'])
plt.xlabel('SibSp class')
plt.ylabel('Survival Rate')
plt.title('SibSp Survival Rate')
plt.show()

# A: We can see that 1-2 siblings/spouses had higher survival rates compared to the others.

# Q: Find the correlations between the feature and the target variable _Survived_ and use heatmap to visualize it.

sns.heatmap(titanic.corr(), annot = True)

# A: From the heatmap generated, we can see that Survived variable is correlated most with Fare, as the correlation value is 0.26. On the other hand, we can conclude that Pclass has the weakest correlation with survival rate with a value of -0.34.

# Q: Any other insights do you draw by analyzing the data?

sns.barplot(data = titanic, x = "Embarked", y = "Survived", hue = 'Sex')

# A: I was interested to see what the survival rate looked like for the passengers from each of the three ports. From the chart generated below, 
# We can see that Cherbourg had the highest survival rate among males and females compared to the other two ports.
