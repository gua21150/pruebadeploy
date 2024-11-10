from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar los modelos PCA y red neuronal
pca_model = joblib.load('pca_model.pkl')
nn_model = joblib.load('nn_model.pkl')

# Columnas
numeric_columns = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", 
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears", 
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole", 
    "YearsSinceLastPromotion", "YearsWithCurrManager"
]
categorical_columns = [
    'BusinessTravel', 'Department', 'Education', 'EducationField', 
    'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 
    'JobRole', 'JobSatisfaction', 'MaritalStatus', 'OverTime', 
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 
    'WorkLifeBalance'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Datos de entrada en formato JSON
    
    # Separar datos numéricos y categóricos
    num_data = pd.DataFrame([data['numerical']], columns=numeric_columns)
    cat_data = pd.DataFrame([data['categorical']], columns=categorical_columns)
    
    # Aplicar transformación PCA a datos numéricos y obtener solo las primeras dos componentes
    num_data_pca = pca_model.transform(num_data)[:, :2]
    
    # Dummificar las variables categóricas
    cat_data_dummies = pd.get_dummies(cat_data)
    
    # Unir datos transformados
    full_data = pd.concat([pd.DataFrame(num_data_pca), cat_data_dummies], axis=1)
    
    # Realizar predicción
    prediction = nn_model.predict(full_data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run()
