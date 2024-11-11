from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Esto permite todas las solicitudes de cualquier origen

# Cargar los modelos PCA y red neuronal
pca_model = joblib.load('pca_model.pkl')
nn_model = load_model('models\\NNmodel.h5')

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

expected_columns = [
    "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely",
    "Department_Human Resources", "Department_Research & Development", "Department_Sales",
    "Education_1", "Education_2", "Education_3", "Education_4", "Education_5",
    "EducationField_Human Resources", "EducationField_Life Sciences", "EducationField_Marketing",
    "EducationField_Medical", "EducationField_Other", "EducationField_Technical Degree",
    "EnvironmentSatisfaction_1", "EnvironmentSatisfaction_2", "EnvironmentSatisfaction_3", "EnvironmentSatisfaction_4",
    "Gender_Female", "Gender_Male", "JobInvolvement_1", "JobInvolvement_2", "JobInvolvement_3", "JobInvolvement_4",
    "JobLevel_1", "JobLevel_2", "JobLevel_3", "JobLevel_4", "JobLevel_5",
    "JobRole_Healthcare Representative", "JobRole_Human Resources", "JobRole_Laboratory Technician",
    "JobRole_Manager", "JobRole_Manufacturing Director", "JobRole_Research Director", 
    "JobRole_Research Scientist", "JobRole_Sales Executive", "JobRole_Sales Representative",
    "JobSatisfaction_1", "JobSatisfaction_2", "JobSatisfaction_3", "JobSatisfaction_4",
    "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single",
    "OverTime_No", "OverTime_Yes", "PerformanceRating_3", "PerformanceRating_4",
    "RelationshipSatisfaction_1", "RelationshipSatisfaction_2", "RelationshipSatisfaction_3", "RelationshipSatisfaction_4",
    "StockOptionLevel_0", "StockOptionLevel_1", "StockOptionLevel_2", "StockOptionLevel_3",
    "WorkLifeBalance_1", "WorkLifeBalance_2", "WorkLifeBalance_3", "WorkLifeBalance_4"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Datos de entrada en formato JSON
    
    # Separar datos numéricos y categóricos
    num_data = pd.DataFrame([data['numerical']], columns=numeric_columns)
    cat_data = pd.DataFrame([data['categorical']], columns=categorical_columns)

    # Aplicar transformación PCA a datos numéricos y obtener solo las primeras dos componentes
    num_data = StandardScaler().fit_transform(num_data) 
    num_data_pca = pca_model.transform(num_data)[:, :2]
    # Obtener las variables numericas 
    cat_data_dummies = pd.DataFrame(False, index=cat_data.index, columns=expected_columns)

    # Asegurarse de que todas las columnas esperadas están presentes en el DataFrame de entrada, igualando la estructura con la que se entrenó el modelo
    for index, row in cat_data.iterrows():
        for col in cat_data.columns:
            column_name = f"{col}_{row[col]}"
            if column_name in expected_columns:
                cat_data_dummies.loc[index, column_name] = True

    # Reordenar las columnas para que coincidan con el modelo de entrenamiento
    cat_data_dummies = cat_data_dummies[expected_columns]

    # Unir datos transformados
    full_data = pd.concat([pd.DataFrame(num_data_pca), cat_data_dummies], axis=1)
    
    # Aplicar una segunda escalizacion a los datos
    full_data = StandardScaler().transform(full_data)
    
    # Realizar predicción
    prediction_prob = nn_model.predict(full_data)  # Esto devuelve una probabilidad
    prediction_class = (prediction_prob[:, 1] >= 0.5).astype(int)  # Convierte en clase
    return jsonify({'prediction': int(prediction_class[0])})

if __name__ == '__main__':
    app.run()