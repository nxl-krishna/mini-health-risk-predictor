# Health Risk Predictor

A Machine Learning application to predict the risk of diabetes using the PIMA Indians Diabetes Dataset.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python src/model_train.py`
3. Run the app: `streamlit run src/app.py`

## Methodology
- Handles missing values (zeros) via Median Imputation.
- Uses Random Forest Classifier for non-linear decision boundaries.
- Input data is standardized using Z-score normalization.
