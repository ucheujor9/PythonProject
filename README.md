# PythonProject
#Data cleaning, #Analysis, #Machine learning
The dataset contains 1500 rows and 8 columns with missing values in the weight column ore cleaning. I have cleaned all columns against the criteria in the dataset table:

booking_id: same as description without missing values (0001 - 1500).
months_as_member: same as description with minimum of 1 month and no missing values.
weight: cleaned by rounding up to 2 decimal places, removed 20 missing values by replacing them with the overall mean of the weight column and made sure the minimum weight was 40.00kg.
days_before: cleaned by removing the string (days) attached to some fields and converting to integer datatype with a minimum of 1 day.
day_of_week: cleaned by changing the long versions to short versions as specified in the description and changed to category datatype.
time: same as description, changed to category datatype.
category: No visible missing values but had 13 fields with '-' as inputs as part of unique values, changed them to 'Unknown' as specified in the description and changed to category datatype.
attended: same as description.
After claening and validation, the dataset remains 1500 rows and 8 columns
From the graph 1, the category of people that had most observations are the people who did not attend the fitness program. The observations as noticed in the graph are not balanced as the number of people who did not attend are more than those that attended with 0-not attended, 1-attended.
The distribution is not normal, it is left skewed as shown in graph 2A as a result of the outlier at 150-160 as seen in graph 2B.
From graph 3, there is a relationship between the number of attendance and the number of months of membershipship; the attendance is more for people who have registered with the program longer than those who just joined the fitness program as clearly seen in Graph 3B.
This dataset has a classification type of machine learning problem. I used a KNeighborClassifier and a DecisonTreeClassifier to predict if members will attend better in future or not. for the data to be used for modelling, I log-transformed the months_as_member column as seen in Graph 3C, dropped some redundant columns and then label encoded the category observations (time with AM & PM, category and day of the week column). The dataset was split in a way to have the same distribution as the original distribution hence the use of stratify.
Baseline Model - KNeighborsClassifier
Comparison Model - DecisionTreeClassifier
I chose KNN and DecisionTreeClassifier because they are easy to perform and interpret without compromising their accuracy and no assumptions is needed for them to perform.
I used accuracy_score to know the percentage accuracy of each model and confusion matrix to get the number of correct predictions.
From the result of the accuracy score and the confusion matrix, although the results are close, I would say the KNN performed better with an accuracy of 77% over the DecisionTreeClassifier with 74%-75% and a correct prediction that 230 members will not attend the fitness program and 60 members will attend.
