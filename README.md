# Projects
The project is to predict the survived people after the titanic ship drowned on the basis of given attributes i.e,passengerId,survival,pclass,sex,Age,sibsp,Parch,ticket,fare,cabin,embarked.
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

The data has been split into two groups:
training set (train.csv)
test set (test1.csv)

predictions is the dataset of final predictions indexed by passengerId 
The project was done by using LogisticRegression with python. Data analysis and data wrangling were made by pandas,numpy,matplotlib,seaborn.
As the project was done in Jupyter notebook 'Titanic_LogisticRegression.ipynb' is the efficient and easily understandable file.
'Titanic_LogisticRegression.py' is file which is converted from 'Titanic_LogisticRegression.ipynb'.
git fork workflow checking
