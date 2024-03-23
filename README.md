# Project-Based Internship: Kalbe Nutritionals x Rakamin Academy

## Introduction

This project was conducted as a final task for the Project-Based Internship with Kalbe Nutritionals x Rakamin Academy from August to September 2023. The project aimed to utilize data analysis and machine learning techniques to provide insights and recommendations for sales performance improvement.

## Video Presentation: 
https://www.youtube.com/watch?v=49MfaclzTj8


## Objectives

- Perform Exploratory Data Analysis (EDA) using DBeaver and PostgreSQL.
- Ingest and preprocess data to create effective data visualizations for product and sales trends analysis.
- Utilize the ARIMA Time-Series Model to forecast and predict daily product sales for inventory management.
- Implement customer segmentation using K-Means Clustering and propose business recommendations to enhance sales performance.

## Tools Used

- DBeaver
- PostgreSQL
- Python (Pandas, Matplotlib, Seaborn, and Scikit-learn)
- ARIMA Time-Series Model
- K-Means Clustering

## Project Structure

The project includes the following tasks:

1. **Exploratory Data Analysis (EDA):** 
   - Utilized DBeaver and PostgreSQL to explore the dataset.
   - Analyzed various aspects of the data to gain insights into sales trends and patterns.

2. **Data Visualization:**
   - Ingested and preprocessed data to create meaningful visualizations.
   - Generated visualizations using Tableau to provide a better understanding of the current sales report.
     
     <img width="600" alt="image" src="https://github.com/alexander-steven/VIX-Kalbe-Nutritionals/assets/74502692/e72a32df-5cd5-43e2-aa10-00a5f53d3a19">
      Link: https://shorturl.at/fwxEO

3. **Time-Series Forecasting:**
   - Implemented the ARIMA Time-Series Model to forecast and predict daily product sales.
   - Utilized the forecasts for effective inventory management.
   - Using AutoArima, ACF-PACF, and Pandas Autocorrelation plot to test most appropriate value for the ARIMA Model paramater (p, d, q)
     
   - Prediction Result:
     <img width="1085" alt="image" src="https://github.com/alexander-steven/VIX-Kalbe-Nutritionals/assets/74502692/ff7be317-8216-4bea-9130-8517f232726e">


4. **Customer Segmentation:**
   - Applied K-Means Clustering to segment customers based on their purchasing behavior.
   - Using the Elbow Method and Silhouette Score to find the most appropriate K Value
   - Proposed business recommendations to the marketing team to enhance sales performance based on the segmentation results.
   - Segmentation Result:
   
     <img width="359" alt="image" src="https://github.com/alexander-steven/VIX-Kalbe-Nutritionals/assets/74502692/4e2d4942-5618-4853-9e3b-8d7ef8048f7e">


## Business Insights

1. In general, this project provided valuable insights into sales trends and customer behavior, enabling informed decision-making and actionable recommendations for sales performance improvement.
2. Time-Series prediction model can be used by the inventory team to check for product stocks for these particular periods of time. Ensuring the items are always available to prevent no-stocks happening.
3. Customer Segmentation can be utilized by the marketing team to find which clusters needs more attention, which one is our loyal customer, and which one's relationship needs to be maintained.
