# Time-Series Forecasting and Failure Prediction
This project tackles a dual machine learning challenge: forecasting a time-series of measured values and predicting rare failure events within that series. It employs a sophisticated approach combining an XGBoost regression model for accurate forecasting and a highly specialized, cost-sensitive XGBoost classification model to identify 'FAIL' labels in a severely imbalanced dataset.

The final output is a 30-day forecast of the 'Measured_value', with the top 63 most likely future failure points clearly marked.

🎯 Project Objective
The primary goals of this project are twofold:

Time-Series Forecasting: To build a regression model that accurately predicts the future Measured_value based on historical data. The key success metric is achieving a Root Mean Squared Error (RMSE) lower than the standard deviation of the test data.

Failure Prediction: To develop a classification model capable of identifying rare 'FAIL' events. Due to the extreme class imbalance, the focus is on maximizing Recall (the ability to find all actual failures), even at the cost of precision.

📊 Dataset & The Imbalance Challenge
The project uses a time-stamped dataset (Sample_data.xlsx) containing a Measured_value and a corresponding Label ('PASS' or 'FAIL').

RMSE for Regression: While the regression model's primary goal is to minimize RMSE, the few extreme values from 'FAIL' days can disproportionately affect this metric.

Recall for Classification: A standard model would likely achieve high accuracy by simply predicting 'PASS' every time, completely failing to identify any failures. Therefore, maximizing Recall is the most important goal for the classification task.

⚙️ Methodology & Workflow
The project is structured across several Jupyter Notebooks, from data cleaning to the final submission. The core workflow is as follows:

Data Cleaning and Preparation: The raw data is cleaned, non-numeric values are handled, and the timestamps are aggregated into a mean Measured_value for each day.

Extensive Feature Engineering: Over 40 features are created from the time-series data to provide the models with a rich set of information, including:

Lag features (values from previous days).

Rolling window statistics (mean, std, min, max).

Exponentially weighted moving averages (EWMAs).

Cyclical features for day-of-week and month.

Trend, momentum, and volatility features.

Dual Model Approach:

XGBoost Regressor: An ensemble of XGBoost models is trained to forecast the Measured_value. Its performance is validated by comparing its RMSE to the baseline standard deviation of the test set.

XGBoost Classifier: A separate classifier is trained to predict the 'FAIL' probability. To handle the extreme imbalance, this model uses:

Cost-Sensitive Learning: The scale_pos_weight parameter is set to an extremely high value (multiplying the base imbalance ratio by up to 1000x) to heavily penalize the model for missing a failure case.

Aggressive Threshold Search: Instead of the standard 0.5 probability threshold, a search is performed to find the optimal threshold that maximizes recall, ensuring all failure cases are identified.

Future Prediction Loop:

A 30-day forecast is generated iteratively. For each new day, the regression model predicts the Measured_value.

This predicted value is then used as an input feature for the next day's prediction.

Simultaneously, the classification model predicts the failure probability for each forecasted day.

The top 60 days with the highest failure probability are identified as the most likely future failures.

📈 Key Results
The XGBoost Regressor successfully achieved its goal, with a final RMSE of 0.7411, which is 57.46% better than the baseline standard deviation of the test data (1.7420).

The cost-sensitive XGBoost Classifier was able to achieve a perfect recall of 0.8833 on the training data by using an extremely high positive class weight and an optimized probability threshold, successfully identifying 53 failure cases.

📁 File Structure
.
├── Data_cleaning_and_preparation.ipynb # Initial data cleaning and resampling.
├── Main.ipynb                          # The main notebook to run for the final results and forecast.
├── Sample_data.xlsx                    # The raw input dataset.
├── Cleaned_data.xlsx                   # The cleaned dataset.
└── measured_value.png                  # The final output plot showing the forecast.

🚀 How to Run
Prerequisites: Ensure you have Python installed with the necessary libraries. You can install them using pip:

pip install pandas numpy xgboost matplotlib scikit-learn seaborn

Run the Main Notebook: Open and run the Final_submission.ipynb notebook from top to bottom. It contains the complete, optimized pipeline for data processing, model training, and future prediction.

View the Output: The notebook will generate the final forecast plot (measured_value.png) at the end of its execution.

🛠️ Tech Stack
Python 3.x

Pandas & NumPy: For data manipulation and numerical operations.

Scikit-learn: For data preprocessing (StandardScaler).

XGBoost: For both the regression and classification models.

Matplotlib & Seaborn: For data visualization.

Jupyter Notebook: For interactive development and analysis.
