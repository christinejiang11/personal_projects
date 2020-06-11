# Personal Projects
Collection of data science projects for fun and self-learning. 

*Primary tools: pandas, numpy, matplotlib, seaborn, sklearn, statsmodels, fbprophet, bokeh, nltk, and pygsheets.*

## Contents
#### Machine Learning
- [MNIST Fashion Recognizer](MNIST_fashion_recognizer/MNIST_Fashion_Recognizer.ipynb): A convolutional neural net to classify 60,000 images of clothing with a final model accuracy of 92%. Includes exploratory models and visual analysis to better understand the effect of filters and kernel size on model training time and results. 
- [Convert to Paid Prediction](conversion_prediction/CTP_Classifier.ipynb): Creating a classifier to predict whether or not free-trial customers on an online learning platform will convert to paid customers at the end of their one-month trial. Explores the differences in model accuracy between three models: K-nearest neighbors, logistic regression, and random forest. 
- [Wikipedia Time Series](wikipedia_time_series/Web_Traffic_Prediction.ipynb): Multiple time series project with the goal of predicting Wikipedia web views from January 2017 to March 2017 for 145,000 different pages. Uses both the statsmodels library ARIMA model and Facebook's Prophet model and compares model results. 

#### Visualization
- [NYC 311 Calls](nyc_311_dashboard/NYC_311_Calls.ipynb): An interactive dashboard to visualize 311 (non-emergency) call volume in New York City. Serves as a proof-of-concept for how analytics can be used to understand fluctuations in call volume and better prioritize resources and case management across the city. To download scripts and run the dashboard yourself, see [here](../NYC_311_Dash/). 

#### Natural Language Processing
- [Whispr Sentiment Analysis](whispr_sentiment_analysis/Sentiment_Analysis.ipynb): Simple sentiment analysis for marketing consulting firm - classifies social media posts as positive, neutral, or negative using nltk and pygsheets.
