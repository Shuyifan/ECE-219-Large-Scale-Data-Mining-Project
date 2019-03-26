# 219-Project
Large-Scale Data Mining: Models and Algorithms

Course description: (Formerly numbered Electrical Engineering 219.) Lecture, four hours; discussion, one hour; outside study, seven hours. Introduction of variety of scalable data modeling tools, both predictive and causal, from different disciplines. Topics include supervised and unsupervised data modeling tools from machine learning, such as support vector machines, different regression engines, different types of regularization and kernel techniques, deep learning, and Bayesian graphical models. Emphasis on techniques to evaluate relative performance of different methods and their applicability. Includes computer projects that explore entire data analysis and modeling cycle: collecting and cleaning large-scale data, deriving predictive and causal models, and evaluating performance of different models. Letter grading.
## Project 1: Classification Analysis on Textual Data
Classification refers to the task of identifying a category, from a predefined set, to which a data point belongs, given a training data set with known category memberships. In this project, we researched different methods for classifying the textual data.

We firstly preprocessed the textual data by tokenizing each document into words, and then transformed it into a document-item matrix, where each row represents a document and each column represents the number of occurrences of a term in each document. Then we further used TF-IDF score to finally determine our data. After that, we performed dimensionality reduction by using Latent Semantic Indexing (LSI) and Non-Negative Matrix Factorization (NMF) methods.

With the data, we applied several different classification algorithms, starting with binary classification. In this project, we used Support Vector Machine (SVM), Naive Bayes, and Logistic Regression under different hyper-parameters. To evaluate how well our algorithm performs, we plotted the ROC curve, calculated the confusion matrix, accuracy, recall, precision and F-1 score.

At last, we performed SVM and Naive Bayes multiclass classification (with both One VS One and One VS the rest methods) and performed the same evaluation metrics for these classification methods.
## Project 2: Clustering
Cluster algorithms are unsupervised methods for finding groups of data points that have similar representations in a feature space. Clustering differs from classification in that no a priori labeling (grouping) of the data points is available.

In this project, we were focusing on a cluster algorithm call K-mean clustering. K-means clustering is a simple and popular cluster algorithm. Given a set of data points {x ⃗_1,...,x ⃗_N}in multidimensional space, it tries to find K clusters s.t. each data point belongs to one and only one cluster, and the sum of the squares of the distances between each data point and the center of the cluster it belongs to is minimized.

During this project, we used the “20 Newsgroup” dataset. We firstly preprocessed the the textual data by tokenizing each document into word, and then transformed it into a document-item matrix. Then, we used TF-IDF score to finally determine our data.

With the transformed data, we explored the K-means algorithm and the evaluation method of a clustering result (contingency matrix, homogeneity score, completeness score, V-measure score, adjusted Rand Index score, and adjusted mutual information score). We also investigated how PCA, scaling, and logarithm transformation can affect the performance clustering result using K-means algorithm. 

At last, we expanded our dataset from 2 categories to 20 categories. Using different combination of PCA, scaling, and logarithm transformation to get a best result.
## Project 3: Collaborative Filtering
The basic idea of recommender systems is to utilize user data to infer customer interests. The entity to which the recommendation is provided is referred to as the user, and the product being recommended is referred to as an item.

The basic models for recommender systems work with two kinds of data:
1.	User-Item interactions such as rating
2.	Attribute information about the users and items such as textual profiles or relevant keywords

Models using first type of data are referred to as collaborative filtering methods, whereas models that use second type of data are referred to as content based methods. In our project, we will build recommendation system using collaborative filtering methods.

One main challenge in designing collaborative filtering method is that the underlying rating matrix is sparse. So the basic idea is that these unspecified ratings can be imputed because the observed ratings are often highly correlated across various users and items. Most of the collaborative filtering methods focus on either inter-item correlation or inter-user correlation for the prediction process.

In this project, we implemented and analyzed the performance of two types of collaborative filtering methods:
1.	Neighborhood-based collaborative filtering
2.	Model-based collaborative filtering

We build a recommendation system to predict the ratings of the movies in the MovieLens dataset. For the subsequent discussion, we assume that the ratings matrix is denoted by R, and it is an $m×n$ matrix containing $m$ users (rows) and $n$ movies (columns). The $(i,j)$ entry of the matrix is the rating of user $i$ for movie $j$ and is denoted by $r_{ij}$.
## Project 4: Regression Analysis
Regression analysis is a statistical procedure for estimating the relationship between a target variable and a set of potentially relevant variables. In this project, we explore basic regression models on a given dataset, along with basic techniques to handle over- fitting; namely cross-validation, and regularization. With cross-validation, we test for overfitting, while with regularization we penalize overly complex models. 
## Project 5: Application - Twitter data
A useful practice in social network analysis is to predict future popularity of a subject or event. During this project, we are using Twitter data to predict what will become popular. We used the available Twitter data collected by querying popular hashtags related to the 2015 Super Bowl spanning a period starting from two weeks before the game to a week after the game. Besides, we used data from some of the related hashtags to train a regression model and then use the modeltomakepredictionsforotherhashtags. We tested several models, using the training data to train the data, and compare the performance of them by using a test data to make predictions.