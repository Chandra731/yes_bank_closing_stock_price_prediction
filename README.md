# Yes Bank Close Stock Price Prediction

This project aims to predict the closing stock prices of Yes Bank using various Machine Learning models. The key steps included data collection, exploratory data analysis (EDA), feature extraction, model training, evaluation, and deployment.

## Project Summary

### Data Collection and Preprocessing

Historical stock price data for Yes Bank was collected, which included features such as the opening price, closing price, highest price, lowest price, and trading volume. Additional derived features such as price change, daily range, year, and month were included to enhance predictive power.

### Model Training and Evaluation

Several Machine Learning models were implemented and evaluated, including ARIMA, Random Forest Regressor, XGBoost Regressor, and Support Vector Regressor (SVR). Each model was trained on the training set and evaluated on the test set using metrics such as MSE, MAE, MAPE, R2, and Adjusted R2.

- **Random Forest Regressor**: Showed good performance with a good R2 score and low error rates. Hyperparameter tuning slightly improved the performance.
- **XGBoost Regressor**: Performed well with good evaluation metrics. Hyperparameter tuning marginally improved the performance.
- **Support Vector Regressor (SVR)**: Demonstrated the best performance among all models, with the lowest error rates and the highest R2 score after hyperparameter tuning.

### Model Saving and Deployment

The best-performing SVR model was saved using joblib for deployment purposes. The model was then loaded and a sanity check was performed by predicting on a new set of unseen data. The model maintained its performance on unseen data, confirming its robustness and readiness for deployment.

## Dataset

The dataset used in this project is available in the `data` directory of the repository.

## Running the Code

1. Clone the repository:
   ```
   git clone https://github.com/Chandra731/yes_bank_closing_stock_price_prediction.git
   cd yes_bank_closing_stock_price_prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```
   jupyter notebook Yes_Bank_Close_Price_Prediction.ipynb
   ```

## Future Work

1. **Feature Engineering**: Incorporate additional features such as economic indicators, news sentiment analysis, and other relevant factors.
2. **Advanced Models**: Explore advanced techniques like deep learning models, ensemble methods, and hybrid models.
3. **Deployment**: Optimize the deployment process for scalability and efficiency, and integrate the model into a web application for real-time predictions.

## Conclusion

This project demonstrated the power of machine learning in stock price prediction, providing a solid foundation for future enhancements and real-world applications. The SVR model's superior performance makes it an excellent choice for predicting Yes Bank's stock prices, aiding investors and financial analysts in making informed decisions.
