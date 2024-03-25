# Constants for the project
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import randint, uniform

DATA_DIR = 'data/'
RAW_DATA_DIR = 'data/raw_data/'
PROCESSED_DATA_DIR = 'data/processed_data/'

CONTINUOUS_COLS = ['age', 'avg_glucose_level', 'bmi']
DISCRETE_COLS = ['hypertension', 'heart_disease', 'stroke']
NOMINAL_COLS = ['gender', 'work_type', 'Residence_type',
                'smoking_status', 'ever_married', 'diabetic_status', 'weight_status']

USED_MODELS = {
    "GradientBoosting": (GradientBoostingClassifier, {
        'n_estimators': randint(50, 250),
        'learning_rate': uniform(0.001, 0.7),
        'max_depth': randint(1, 20),
        'min_samples_leaf': randint(1, 20)}),
    "KNN": (KNeighborsClassifier, {
        'n_neighbors': randint(1, 100),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),
    "RandomForest": (RandomForestClassifier, {
        'n_estimators': randint(2, 200),
        'max_depth': randint(1, 10)}),
    "SVM": (SVC, {
        'C': uniform(0.01, 10),
        'kernel': ['rbf', 'linear']})
}

# utility functions

def _create_weight_status(data):
    if data < 18.5:
        return 'underweight'
    elif 18.5 <= data < 24.9:
        return 'normal'
    elif 25.0 <= data < 29.9:
        return 'overweight'
    else:
        return 'obese'


# research: https://www.cdc.gov/diabetes/basics/getting-tested.html
def _create_diabetic_status(data):
    if data <= 140:
        return 'normal'
    elif 140 < data <= 200:
        return 'prediabetic'
    else:
        return 'diabetic'


def create_new_features(df):
    """Adds two new features to the dataframe. The two new features are 'weight_status' and 'diabetic_status'.

    Args:
        df (pandas.DataFrame): The dataframe to add the new features to.

    :returns: The dataframe with the new features.
        """
    df['weight_status'] = df['bmi'].apply(_create_weight_status)
    df['diabetic_status'] = df['avg_glucose_level'].apply(_create_diabetic_status)
    return df

