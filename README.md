# Titanic Data Analysis


### Overview
In this Python file, I utilized pandas, matpotlib, and seaborn to analyze and visualize the Titanic dataset. Specifically, I wanted to see which factors impacted the passengers' survival rate the most, as well as the least. Furthermore, I used various visualization techniques to create graphs of the data, including heatmaps, distribution plots, and bar plots.

## Dataset
https://raw.githubusercontent.com/zariable/data/master/titanic_train.csv. 

### Dataset Factors
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

### Conclusion
From my analysis, I drew that passenger survival rate was heavily dependant on their fare. As seen in the generated heatmap below, the correlation factor between survival and fare is 0.26, indicating that they are strongly related. On the other hand, Pclass is the weakest correlated factor (-0.34) to survival.

![image](https://user-images.githubusercontent.com/63205351/231035649-effe4dbe-857f-472c-bd43-3066e73b040f.png)

It was interesting to see that survival was highest for those who had 1-3 parents/children, where as 0, 4, 5, and 6 parents/children had lower survival rates.

![image](https://user-images.githubusercontent.com/63205351/231038129-deb58b70-2f14-451c-be61-4ce19965ae86.png)

We can see that 1-2 siblings/spouses had higher survival rates compared to the others.

![image](https://user-images.githubusercontent.com/63205351/231038102-8ce27020-4c54-4689-acb2-545540381dca.png)

Lastly, I was interested to see what the survival rate looked like for the passengers from each of the three ports. From the chart generated below, we can see that Cherbourg had the highest survival rate among males and females compared to the other two ports.

![image](https://user-images.githubusercontent.com/63205351/231038027-ca002a59-4639-4c8c-bf59-2201901508b3.png)

