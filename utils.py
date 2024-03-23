# Constants for the project

DATA_DIR = 'data/'
RAW_DATA_DIR = 'data/raw_data/'
PROCESSED_DATA_DIR = 'data/processed_data/'

CONTINUOUS_COLS = ['age', 'avg_glucose_level', 'bmi']
DISCRETE_COLS = ['hypertension', 'heart_disease', 'stroke']
NOMINAL_COLS = ['gender', 'work_type', 'Residence_type',
                'smoking_status', 'ever_married', 'diabetic_status', 'weight_status']


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

