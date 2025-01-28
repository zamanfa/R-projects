# I will analyse the Boston Housing dataset. The dataset is used for predicting house prices based on various features such as crime rate, average number of rooms and property tax rates. I will aim to predict a continuous target variable: the median house value.

library(MASS)
library(ISLR)
library(caret)

data(Boston)
head(Boston)

summary(Boston)

# The Boston Housing Dataset contains information about different housing attributes for 506 observations in the Boston area. The dataset has 14 variables, with 13 predictor variables and 1 target variable (median house value).

# Variables in the Dataset: 
# crim: Crime rate per capita by town.
# zn: Proportion of residential land zoned for large lots.
# indus: Proportion of non-retail business acres per town.
# chas: Charles River dummy variable (1 if the tract bounds the river; 0 otherwise).
# nox: Nitrogen oxide concentration (parts per 10 million).
# rm: Average number of rooms per dwelling.
# age: Proportion of owner-occupied units built before 1940.
# dis: Weighted distance to employment centers.
# rad: Index of accessibility to radial highways.
# tax: Property tax rate per $10,000.
# ptratio: Pupil-teacher ratio by town.
# black: Proportion of African American residents by town.
# lstat: Percentage of lower status population.
# medv (target variable): Median value of owner-occupied homes in $1,000s.

# The goal is to predict medv, the median house value, based on the features of the houses and neighborhoods.


set.seed(100)
trainIndex = createDataPartition(Boston$medv, p = 0.8, list = FALSE)
trainData = Boston[trainIndex, ]
testData = Boston[-trainIndex, ]

# Supervised Learning Model 1: Linear Regression

# I will start with Linear Regression. The model will predict the target variable (medv) based on the 13 predictor variables.

# Model Formula:
# The formula for the linear regression model will include all the predictor variables as independent variables:
  
# medv = β0 + β1(crim) + β2(zn) + β3(indus) + ⋯ + β13(lstat) + 𝜖 
# Where:
# medv is the target variable (median house value),
# crim, zn, indus, etc., are the predictor variables,
# β0 is the intercept term,
# β1, β2, ..., β13 are the coefficients (weights) for each predictor,
# ε is the error term (residuals).

lm.fit = lm(medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, data = trainData)

summary(lm.fit)

# Residuals:

# The residuals represent the differences between the observed and predicted values. Here's how to interpret them:

# Min: -14.08
# 1st Quartile (1Q): -2.75
# Median: -0.45
# 3rd Quartile (3Q): 1.63
# Max: 24.78

# These values tell us that most residuals are close to 0, but there are some relatively large residuals (both positive and negative), suggesting the model may not fully capture all the variance in the data. The spread indicates that there is variability in the errors.

# Coefficients:

# The coefficients represent the change in the predicted median house value (medv) for each one-unit increase in the corresponding predictor variable, assuming all other variables are held constant.

# Here are some key interpretations:
  
# (Intercept): 41.16. This is the estimated median house value when all predictor variables are zero. While this value might not be meaningful on its own (because it's unlikely that all predictors would be zero), it serves as a starting point for the model.
# crim: -0.0862. As the crime rate increases by one unit, the median house value decreases by approximately $86 (since the target variable is in thousands of dollars).
# zn: 0.0646. As the proportion of residential land zoned for large lots increases by one unit, the median house value increases by approximately $64.
# chas: 2.8225. If the property is near the Charles River (i.e., chas = 1), the median house value increases by approximately $2,822 compared to properties that are not near the river.
# rm: 3.0220. For each additional room in the house, the median house value increases by $3,022. This is one of the most important predictors for house value in this model, as we would expect.
# nox: -16.6081. As the nitrogen oxide concentration increases by one unit, the median house value decreases by approximately $16,608.
# lstat: -0.6008. As the percentage of lower-status population increases by one unit, the median house value decreases by approximately $600.

# Statistical Significance (Pr(>|t|)):

# Significance Codes: The Pr(>|t|) values indicate the probability that the predictor is not statistically significant (i.e., that the coefficient is zero).
# Significant predictors (p < 0.05): Many predictors, including crim, zn, chas, nox, rm, dis, rad, tax, ptratio, black, and lstat, have p-values less than 0.05, suggesting they are statistically significant in predicting house prices.
# Insignificant predictors (p > 0.05): indus and age have p-values greater than 0.05, suggesting that they are not statistically significant in predicting median house values in this model.

# R-squared:

# Multiple R-squared: 0.7422. This means that 74.22% of the variability in the median house value is explained by the model. This is a relatively strong fit for the data.
# Adjusted R-squared: 0.7337. This value is slightly lower than the Multiple R-squared because it adjusts for the number of predictors. It is still quite good and suggests that the model explains a large portion of the variability in house prices.

# F-statistic:

# F-statistic: 87.02 with 13 and 393 degrees of freedom. This is a measure of how well the model fits the data. The high F-statistic and a p-value < 2.2e-16 indicate that the model is statistically significant overall, meaning at least some of the predictors are useful for predicting house values.

# Residual Standard Error:

# Residual Standard Error: 4.753. This is the average distance that the observed values fall from the regression line. A smaller value indicates a better fit, but the value of 4.753 suggests there’s still considerable variability in the data that the model hasn’t fully captured.

lm_predictions = predict(lm.fit, newdata = testData)
lm_mse = mean((lm_predictions - testData$medv)^2)  
lm_rmse = sqrt(lm_mse) 
lm_rmse

library(rpart)
library(rpart.plot)

# Supervised Learning Model 2: Decision Tree

#Next, I will build  a Decision Tree to predict the median house value. Decision trees are useful for capturing non-linear relationships between the predictors and the target variable.
# I will use the same formula as the linear regression model but apply it to a decision tree model.

tree.fit = rpart(medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, data = trainData, method = "anova")
rpart.plot(tree.fit)

# Key Splits in the Tree:

# Root Node (lstat >= 9.8)
# The first and most important split is lstat. This indicates that lstat is the most influential predictor in the decision tree model.
# If lstat >= 9.8, we move left.
# If lstat < 9.8, we move right.

# Left Subtree (Higher lstat values, meaning lower-income areas)
# Further splits occur based on lstat >= 16, indicating another threshold for the lower-status population affecting house prices.
# dis < 1.9 appears as a split within this group, meaning that distance to employment centers is relevant for pricing.
# Houses with very high lstat and small dis values tend to have the lowest house prices (e.g., terminal node at medv ≈ 11).

# Right Subtree (Lower lstat, meaning higher-income areas)
# The next major split is rm < 6.9. This makes sense because the number of rooms (rm) is a key indicator of house value.
# Further, the tree splits based on age < 89, meaning the age of homes plays a role in pricing.
# Houses with more rooms (rm ≥ 6.9) tend to have higher prices, while houses with fewer rooms tend to have lower prices.
# Pupil-Teacher Ratio (ptratio >= 17) appears in the deeper splits, meaning that areas with high student-to-teacher ratios may have lower house prices.

tree_predictions = predict(tree.fit, newdata = testData)
tree_mse = mean((tree_predictions - testData$medv)^2)  
tree_rmse = sqrt(tree_mse)  
tree_rmse

lm_rmse
tree_rmse

# RMSE (Root Mean Squared Error) is a commonly used metric to evaluate regression models. It represents the average magnitude of prediction errors, where lower values indicate better predictive accuracy.

# Linear Regression Model RMSE: 4.89324
# Decision Tree Model RMSE: 4.639082

# Interpretation:

#Lower RMSE for Decision Tree:
# The decision tree model has a slightly lower RMSE (4.639) compared to the linear regression model (4.893). This suggests that the decision tree is marginally better at predicting house prices in this dataset.

# Practical Implications:

# The difference in RMSE is 0.254. While this difference is relatively small, it indicates that the decision tree can capture more complex, non-linear relationships in the data compared to the linear regression model, which assumes a linear relationship between predictors and the target variable.

# Model Selection:

# The decision tree may be more suitable for this dataset if the goal is purely to minimize prediction errors. However, linear regression offers more interpretability, allowing us to understand how each predictor affects the target variable.
# If explainability is crucial (e.g., understanding the relationships between features like crime rate, number of rooms, and house price), the linear regression model may still be preferred despite its slightly higher RMSE.