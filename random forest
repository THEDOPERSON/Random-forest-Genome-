import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r'C:\Users\ASUS\Downloads\ML_Q1DA\METABRIC_RNA_Mutation.csv')

# Encode categorical features
label_encoders = {}
for column in ['type_of_breast_surgery', 'cancer_type_detailed', 'cellularity', 'pam50_+_claudin-low_subtype']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
X = data[['age_at_diagnosis', 'type_of_breast_surgery', 'cancer_type_detailed', 'cellularity', 'chemotherapy']]
y = data['pam50_+_claudin-low_subtype']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print accuracy and classification report
print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nClassification Report:\n', report)

# Feature importance
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f'{feature}: {importance:.2f}')
