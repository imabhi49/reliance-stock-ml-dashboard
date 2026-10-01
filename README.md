## Methodology Summary

Historical Reliance and NIFTY 50 data were collected and converted into technical and market-context features.

The models were trained using chronological data splitting to prevent future-data leakage.

The final workflow was:

1. Data collection
2. Data cleaning and exploratory analysis
3. Feature engineering
4. Chronological splitting
5. Logistic Regression
6. Random Forest
7. XGBoost
8. LSTM
9. Model comparison
10. Walk-forward evaluation
11. Feature refinement
12. NIFTY market-context integration
13. Final unseen-test evaluation
14. Backtesting
15. Streamlit deployment

The strongest walk-forward configuration was the market-context Random Forest.

Final unseen-test ROC-AUC was 0.5267, indicating weak but positive directional discrimination.

The more notable result was obtained when predictions were used selectively rather than on every trading day.
