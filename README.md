# Rainfall Prediction Using Ensemble Methods

This project involves developing a machine learning model to predict rainfall in Sydney. The model utilizes various classification techniques, including decision trees and ensemble methods such as Random Forest and Gradient Boosting. The project encompasses data preprocessing, model training, evaluation, and comparison to identify the best-performing model. This work was completed as part of an internship preparation course on Data Science provided by Internshala Trainings.

## Project Overview

### Problem Statement
The Daily Buzz, a small newspaper company in Sydney, aims to attract more readers by providing accurate weather forecasts through a new column, "The Weather Oracle." The editor-in-chief decided to leverage machine learning to improve rainfall prediction accuracy and hired you as the ML expert for this task. You are required to create an ML model using various classification models to predict rainfall.

### Dataset
The dataset contains weather information of Sydney from 2008 to 2017, with 18 columns:
- `Date`: The date of observation
- `Location`: The common name of the location of the weather station
- `MinTemp`: The minimum temperature in degrees Celsius
- `MaxTemp`: The maximum temperature in degrees Celsius
- `Rainfall`: The amount of rainfall recorded for the day in mm
- `Evaporation`: The Class A pan evaporation (mm) in the 24 hours to 9am
- `Sunshine`: The number of hours of bright sunshine in the day
- `Humidity9am`: Humidity (percent) at 9am
- `Humidity3pm`: Humidity (percent) at 3pm
- `Pressure9am`: Atmospheric pressure (hpa) reduced to mean sea level at 9am
- `Pressure3pm`: Atmospheric pressure (hpa) reduced to mean sea level at 3pm
- `Cloud9am`: Fraction of sky obscured by cloud at 9am (measured in oktas)
- `Cloud3pm`: Fraction of sky obscured by cloud at 3pm (measured in oktas)
- `Temp9am`: Temperature (degrees C) at 9am
- `Temp3pm`: Temperature (degrees C) at 3pm
- `RainToday`: Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
- `RainTomorrow`: Boolean: 1 if it will rain tomorrow, otherwise 0 (target variable)

### Approach
1. **Data Loading and Preprocessing**: Load the dataset, handle missing values, encode categorical variables, and normalize the data.
2. **Model Training and Evaluation**: Train various models including Decision Trees, Random Forest, and Gradient Boosting. Evaluate each model's accuracy and create confusion matrices.
3. **Model Comparison**: Compare the performance of the models to identify the best one.
4. **Model Improvement**: Suggest possible ways to further improve the selected model's performance.

## My Task
As an ML expert at The Daily Buzz, you are given the task to create a ML model to predict the rainfall. So, you have to create a Machine Learning Model using various Classification Models including Decision Trees and Ensemble methods, and compare the accuracy of each model. First, load the data and perform data preprocessing and after data cleaning use decision tree classification and then use Bagging and Boosting techniques along with the Random Forest Classifier then find out the accuracy score of each and create a confusion matrix to evaluate the performance. After completing this, take your best model and write why this model performed better than other models and in what ways you can further improve the accuracy of the selected model.

## Questions Answered
1. **Your views about the problem statement?**
   - The problem statement given by "The Daily Buzz" is an example of a classification problem. The company was founded many years ago and is now struggling to attract more readers. To get more readers, they need to innovate and stay ahead of the competition. To address this, they will start a new column called "The Weather Oracle" to predict the weather for the coming days. Predicting rainfall accurately is a classic challenge in meteorology. So, they need me to use historical weather data and advanced machine learning techniques to build models that help the company attract more viewers. The community will be interested in this because they can plan their activities, from daily commutes to agricultural tasks. I have been hired as an ML expert, and they want me to create an ML model to accurately predict the rainfall in Sydney. I will focus on precision (true negatives) because I believe the most important aspect is predicting if it will not rain today so that people can carry out their daily routines, go to the beach, and so on. If the model predicts rain and it does not rain, it is not a big problem. However, it is a bigger problem if the model predicts that it will not rain and it does rain, as this can lead to people being unprepared for adverse weather, disrupting their plans and potentially causing more significant inconvenience or harm.

2. **What will be your approach to solving this task?**
   - First, I will import the libraries and some functions that I will need further. I will import the data and I will make data processing following these steps:

     1. Data verification and visualization: I will use methods such as describe(), df.info, df.shape, and df.isnull to verify the data, check for null values, and identify discrepancies in the data.
     2. Replacing Missing Values: For columns with a high number of null values, I will fill them with the median to maintain data consistency without introducing bias. For categorical data, I will fill missing values with the mode. For columns with a lower number of null values, I will fill them with the mean since these values are less likely to interfere with the data significantly.
     3. Dropping unnecessary columns: After checking the data, I will identify and drop columns that are not needed, such as 'date' and 'location'.
     4. Creating Dummies: I will create dummy variables using pd.get_dummies with ('drop_first = true') for categorical data because many machine learning algorithms require numerical input, and creating dummies allows us to convert categorical data into a number format that can be provided to these algorithms.
     5. Correlation and data redundancy: I will perform a correlation analysis to understand the relationships between different features. This will help me identify which features are most strongly related to the target variable (rainfall prediction) and each other. I will use methods such as df.corr() to calculate the correlation matrix and visualize it with a heatmap. I will drop columns to avoid data redundancy and multicollinearity and also select the most relevant features for the model.
     6. Plotting scatter and box plots: I will create scatter plots and box plots to visualize the data distributions and relationships between features. Scatter plots will help identify patterns and relationships between numerical variables, while box plots will help detect the presence of outliers and understand the spread and central tendency of the data.
     7. Handling outliers: I will identify and handle outliers in the data. Outliers can skew the results and affect the performance of the machine learning model. I will use the percentile method to detect outliers and decide whether to remove them or transform them to minimize their impact on the analysis.

     After completing data processing, I'll split the data into training and testing sets. Then, for models requiring scaled data, I will standardize it using Standard Scaler. And now I can perform my models such as Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, and Classification Trees.

     I will proceed with building models using ensemble and boosting techniques. First, I will construct the models, then I will search for the best hyperparameters using either random search or grid search. Once the best hyperparameters are identified, I will fit the model and plot the results, including a graph for each parameter and a table summarizing the results of the grid search. Then, I will assess the model's performance using accuracy, precision, ROC AUC, and confusion matrix for both the training and testing data. This comprehensive evaluation will provide insights into how well the model generalizes to unseen data and its predictive capabilities.

     For the model that can perform better, I will conduct a deep search for each hyperparameter to find the best fit, aiming for the highest precision while minimizing changes to the average accuracy. I will prioritize precision because accurately predicting non-rainy days is crucial. However, I don't want the model to drastically decrease its accuracy. I will plot graphs, tables, and scores to visualize and evaluate the results comprehensively.

3. **What were the available ML model options you had to perform this task?**
   - As I mentioned, the task is a classification problem, so I have the following available options for models: Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, and Classification Trees. For ensemble techniques, I have Random Forest, AdaBoost, Gradient Boost, and XGBoost. For hyperparameter tuning, I will use Grid Search or Random Search to find the best parameters.

4. **Which model’s performance is best and what could be the possible reason for that?**
   - All of the models perform well, but the XGBoost Random Forest stands out with the best performance. Therefore, I conducted a deep search for each hyperparameter to find the best fit aiming for the highest precision while minimizing changes to the average accuracy. This process involved an extensive grid search and cross-validation to meticulously explore a wide range of hyperparameter values. The model achieved the best scores through this thorough optimization.

     I believe this success is due to the model's combination of the strengths of random forests and XGBoost. The random forest component reduces overfitting and improves generalization by averaging multiple decision trees. Meanwhile, the XGBoost boosting
