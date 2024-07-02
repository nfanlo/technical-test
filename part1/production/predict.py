import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def sin_transformer(period):
    return lambda x: np.sin(2 * np.pi * x / period)

def cos_transformer(period):
    return lambda x: np.cos(2 * np.pi * x / period)

def preprocess_column(df, column_name):
    values = df[column_name].unique()
    if len(values) == 2:
        df[column_name] = df[column_name].map({values[0]: 0, values[1]: 1})
    elif len(values) == 3:
        df = pd.get_dummies(df, columns=[column_name])
    else:
        df[column_name], uniques = pd.factorize(df[column_name].astype(str))
        period = len(uniques)
        df[f'{column_name}_sin'] = sin_transformer(period)(df[column_name])
        df[f'{column_name}_cos'] = cos_transformer(period)(df[column_name])
        df = df.drop(columns=[column_name])
    return df

def preprocess_data(df):
    for column in df.columns:
        df = preprocess_column(df, column)
    return df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

def main(model_choice, input_data):
    model_path = '/app/models/voting_classifier_hard.pkl'

    if model_choice != 'VotingHard':
        print('Invalid model choice. Please select "VotingHard".')
        return

    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print(f'Model file not found at {model_path}')
        return

    try:
        data = pd.read_csv(input_data)
    except FileNotFoundError:
        print(f'Input data file not found: {input_data}')
        return

    data = preprocess_data(data)
    data_scaled = StandardScaler().fit_transform(data)
    predictions = model.predict(data_scaled)

    print(f'Predictions: {predictions}')

    output_file = 'predictions.csv'
    pd.DataFrame(predictions, columns=['Prediction']).to_csv(output_file, index=False)
    print(f'Predictions saved to {output_file}')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python predict.py <model_choice> <input_data>')
        sys.exit(1)

    model_choice = sys.argv[1]
    input_data = sys.argv[2]
    main(model_choice, input_data)
