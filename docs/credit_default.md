## Data Source

In this project, I have conducted a statistical analysis on credit card dataset. The full datasets are originally provided by Lichman, M. (2013). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) Irvine, CA: University of California, School of Information and Computer Science.

Within the datasets, user information has been collected under a scale of 7067 customers. There are 24 variables, including demographic factors, credit data, history of payment, and bill statements of credit card customers from April 2005 to September 2005, as well as information on the final outcome: did the customer default his/her payment next month or not?

## Methodology

#### Missing values

In order to observe the best presentation of the datasets, I decided to first add a `ID` column as "Customer ID" to both training and testing datasets. The column would help identify each row entry as a unique value and make it easier to re-split the dataset into training and testing datasets after data manipulation.

```R
# Fisrt we want to load the datasets into R environment:
CardT <- readRDS("CardT.rds")
CardV <- readRDS("CardV.rds")
CardT$ID <- c(1:5000) # adding customer ID to the dataset
CardV$ID <- c(5001:7067)

Card <- rbind(CardT,CardV)
str(Card)
summary(Card)
```

After initial EDA, I have found that both missing values are categorical. Hence, I chose mosaic plots as a diagnostic tool to visualise the distribution of those missing values across the dataset `Card`. All missing values of both variables were considered as MCAR (Missing Completely at Random) and removed them from the dataset.

```R
# removing the missing values
Card <- Card %>%
  as_tibble() %>%
  filter(!is.na(EDUCATION)) %>%
  filter(!is.na(MARRIAGE))
```

#### Spearson correlation

By looking at the high collinearity existed among the `BILL_AMT` group, I noticed that this group variables are strongly correlated between each other. 

```R
# Checking for correlations among the variables:
ggcorr(Card[, c(1,5,12:24)], palette = "RdYlGn",
      label = FALSE, label_color = "black")
```

This makes sense since that each variable in the group represents the amount of bill statement at a given time, due to the fact that the amount was recorded each month from April 2005 to September 2005.

#### Feature engineering

Based on the initial explanatory data analysis, evidence was found that client who has delayed his/her payment for three months and above is very likely to default the payment. Also the amount of records with delay length longer than 3 months is also very limited. To improve this limitation and to simplify the model parameters, a delay length of four to nine months or above will all be classified as over three month delay length during the model fitting process.

```R
Card_new$PAY_1 <- rockchalk::combineLevels(Card_new$PAY_1, levs =
    c("4","5","6","7","8"), newLabel = ">3")
Card_new$PAY_2 <- rockchalk::combineLevels(Card_new$PAY_2, levs =
    c("4","5","6","7","8"), newLabel = ">3")
Card_new$PAY_3 <- rockchalk::combineLevels(Card_new$PAY_3, levs =
    c("4","5","6","7","8"), newLabel = ">3")
Card_new$PAY_4 <- rockchalk::combineLevels(Card_new$PAY_4, levs =
    c("4","5","6","7","8"), newLabel = ">3")
Card_new$PAY_5 <- rockchalk::combineLevels(Card_new$PAY_5, levs =
    c("4","5","6","7","8"), newLabel = ">3")
Card_new$PAY_6 <- rockchalk::combineLevels(Card_new$PAY_6, levs =
    c("4","5","6","7","8"), newLabel = ">3")
```

Also since there is no significant difference between "married" and "single" categories (`MARRIAGE`), I decided to combine these two labels into one category "married_or_single”.

```R
Card_new$MARRIAGE <- rockchalk::combineLevels(Card_new$MARRIAGE, levs = c("married",
    "single"), newLabel = "married_or_single")
```

#### Model fitting

The model used in this project is logistic regression model. By adding a **cut-off line of 0.2** for the predicted probability of default to make a binary prediction 'likely to default' or 'unlikely to default'.  From there, a confusion matrix can be computed using R package `caret`. The matrix was computed with a given accuracy of 0.8362.

```R
# setting up a cut-off line
prediction.rd <- ifelse(predictions < 0.2,"unlikely to default","likely to default")
# modify the level of default in testing dataset.
CardV_new$default <- as.factor(CardV_new$default)
levels(CardV_new$default) <- c("unlikely to default","likely to default")
pred <- ordered(prediction.rd, levels = c("unlikely to default","likely to default"))
actual <- ordered(CardV_new$default, levels = c("unlikely to default","likely to
default"))

cm <- table(Predicted = pred, Actual = actual)
library(caret)
confusionMatrix(cm)
```

Then the false positive and false negative rates were calculated using the validation data to produce a ROC chart. The correponding ROC curve was also plotted to visually indicator model's performance. 

```R
# ROC curve
library(ROCR)
Predict_Obj <- prediction(predictions,CardV_new$default)
Perform_Obj <- performance(Predict_Obj, "tpr","fpr")
# Plotting ROC curve
plot(Perform_Obj,main = "ROC Curve",colorize=TRUE,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")
```

Based on the diagrams, it is suggested that even the predicted number of default was approximately similar to the actual outcomes, the model is still lack of certain degree of predictive power in terms of binary classification.
