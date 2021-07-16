import utilities.utilities as pu
from utilities.random_forest import RandomForestClassifier
import os

def train():
    # Parameters
    params = {'feature_set': 0}
    feature_set = params.pop('feature_set')

    # Load data
    data_path = os.path.realpath('./data')
    data, y, data_eval, y_eval, feature_names = pu.load_aug_data(data_path)
    feature_sets, feature_sets_names = pu.variable_sets()

    # Subset data
    X = data[:,feature_sets[feature_set]]
    X_eval = data_eval[:,feature_sets[feature_set]]
    
    # Train and evaluation
    rf = RandomForestClassifier(**params).fit(X, y)
    hit_rate = rf.score(X_eval, y_eval)

    pu.dump_model(rf, 'caravan_classifier')
    print('mean_recall:', hit_rate)
     
if __name__ == '__main__':
    train()