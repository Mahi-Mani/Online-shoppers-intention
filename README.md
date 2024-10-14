# Online Shoppers Intention

## Summary
A machine learning model to predict online shoppers' purchasing intentions using the Online Shoppers Intention dataset. Preprocessed data, handled missing values, and applied feature engineering to improve model performance. Experimented with various classification algorithms to identify patterns in user behavior. Evaluated model accuracy and performance providing insights into customer behavior and improving targeted marketing strategies.

## Data Description
Sourced the dataset from the UCI Machine Learning Repository, comprising feature vectors
for 12,330 sessions, each representing a unique user over a 1-year period. The data has been
carefully curated to prevent bias towards particular campaigns, special days, user profiles, or
time periods.

> ### Numeric Features:

* Administrative :The number of pages of this type (administrative) visited by the user in
that session.

* Administrative_Duration : The total amount of time (in seconds) spent by the user on
administrative pages during the session.

* Informational: The number of informational pages visited by the user in that session.
* Informational_Duration : The total time spent by the user on informational pages.
* ProductRelated : The number of product-related pages visited by the user.
* ProductRelated_Duration : The total time spent by the user on product-related pages.
* BounceRates : The average bounce rate of the pages visited by the user. The bounce rate
is the percentage of visitors who navigate away from the site after viewing only one page.
* ExitRates : The average exit rate of the pages visited by the user. The exit rate is a metric
that shows the percentage of exits from a page.
* PageValues : The average value of the pages visited by the user. This metric is often used
as an indicator of how valuable a page is in terms of generating revenue.
* SpecialDay : This indicates the closeness of the site visiting time to a specific special day
(e.g., Mother’s Day, Valentine's Day) in which the sessions are more likely to be finalized
with a transaction.

> ### Categorical Features:
* Month : The month of the year in which the session occurred - ordinal.
* OperatingSystems: The operating system used by the user - nominal.
* Browser : The browser used by the user - nominal.
* Region : The region from which the user is accessing the website - nominal.
* TrafficType : The type of traffic (e.g., direct, paid search, organic search, referral)
-nominal.
* VisitorType : A categorization of users (e.g., Returning Visitor, New Visitor) - nominal.
* Weekend : A boolean indicating whether the session occurred on a weekend.
Target Variable:
* Revenue : A binary variable indicating whether the session ended in a purchase

## Feature Engineering:
We enriched the dataset by creating several new features:

1. `Seasons`: A categorical (nominal) feature based on the month column, allowing business
owners to analyze revenue trends based on seasons.
2. `Total Pages Visited`: A numeric column indicating the total number of pages visited by a
user, providing deeper insights into user behavior.
3. `Total Duration`: A numeric column representing the total time spent by the user on the
site, derived from the sum of administrative, informational, and product-related durations.
4. `Duration Category`: A categorical (ordinal) feature categorizing the total duration as short,
medium, or long, helping to understand user engagement levels.
5. `Page Values per Product Duration and View`: Two numeric columns offering insights into
the value of pages based on the duration and number of views per product.

These new features increase the total number of features by 5, with 2 categorical and 3 numeric
features, providing richer insights for analysis and decision-making for business owners

## Models Considered
> ### SVM: 

Support Vector Machines
SVM is well-suited for binary classification tasks and since the dataset involves predicting
whether a user will make a purchase or not, we decided to include SVM as one of our models.

> ### Random Forest:

We found that the dataset is imbalanced. We know that random forests can perform well with
unbalanced datasets. So we chose a random forest as one of our models.

> ### Decision Trees:

We chose decision trees for the following two reasons. Decision trees require little data
preparation, such as normalization or scaling, compared to other algorithms, making them easy
to use and implement. Decision trees can provide information about the relative importance of
different features, helping to identify the key factors that influence online shopping intentions.

> ### Gradient Boosting Machine:

We wanted to include ensemble methods that combine the predictions of multiple weak learners
(like decision trees), which often leads to better generalization performance compared to
individual models.

## Conclusion

Based on our analysis of the Online Shopping Intention dataset, we can conclude the following
findings:

1. The dataset contains a `mix of categorical and numeric features`, making it suitable for
machine learning models that can handle both types of data.
2. We explored various machine learning models, including Support Vector Machines
(SVM), Random Forest, Decision Trees, and Gradient Boosting Machines (GBM), to
predict online shopping intentions.
3. Our model demonstrates significant improvement over random guessing. With the
implementation of a `Random Forest classifier, an accuracy of approximately
94% and a recall of 93%` was acheived`. These results indicate that our model is effective in accurately
predicting online shopping intentions, outperforming random guessing by a substantial
margin.
4. The most important features for predicting online shopping intentions were found to be
`pageValues` and `pageValues_per_product_view`. These values indicate how strongly these
features contribute to predicting whether a user will complete a transaction or not.

## Learning Points

A key takeaway from this project is the critical role of feature engineering and model
selection in machine learning. Through meticulous feature selection and engineering,
alongside thoughtful model selection,  achieved high performance in predicting online
shopping intentions. This underscores the significance of these processes in developing
effective machine learning models.

## Author Links
[LinkedIn](https://www.linkedin.com/in/mahisha-gunasekaran)

[GitHub](https://github.com/Mahi-Mani)
