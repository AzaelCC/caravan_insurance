import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from itertools import chain, combinations
from sklearn.metrics import confusion_matrix

import pickle

def load_aug_data(data_path):
    """
    Load augmented data. Including modified variables
    """
    names = np.array(['X' + str(i+1) for i in range(86)]) # Hace una lista de nombres de las variables que van desde X1 hasta X86
    
    data_file = os.path.join(data_path, 'ticdata2000.txt')
    x_eval_file = os.path.join(data_path, 'ticeval2000.txt')
    y_eval_file = os.path.join(data_path, 'tictgts2000.txt')

    data = pd.read_csv(data_file, delimiter="\t",header=None, names=names)
    X_eval = pd.read_csv(x_eval_file, delimiter="\t",header=None, names=names[:-1])
    y_eval = pd.read_csv(y_eval_file, delimiter="\t",header=None, names=['X86'])

    X_train = data.iloc[:,0:85]  # Se seleccionan todas las variables a excepción de la variable respuesta que es la última
    y_train = data.iloc[:, 85] # Se seleccion unicamente la variable respuesta
    
    vars_to_change = [9,18,22,23,29,31,35,36]
    new_vars = np.array(['X'+str(i+1)+'_2' for i in vars_to_change])
    names = np.concatenate((names[:-1], new_vars))
    class_reducer = {0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:4, 7:4, 8:5, 9:5}

    # Add new columns to X_train
    X_train = pd.concat([X_train, X_train.iloc[:,vars_to_change]], axis=1) # Se copian las columnas que vamos a cambiar al final del dataframe
    X_train.columns = names

    # Reduce 10 classes to 5
    for i in range(len(vars_to_change)):
        X_train.iloc[:,i+85] = X_train.iloc[:,i+85].replace(class_reducer)

    # Same for X_eval
    X_eval = pd.concat([X_eval, X_eval.iloc[:,vars_to_change]], axis=1) # Copy columns to mutate
    X_eval.columns = names

    # Reduce 10 classes to 5
    for i in range(len(vars_to_change)):
        X_eval.iloc[:,i+85] = X_eval.iloc[:,i+85].replace(class_reducer)
        
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)
    
    oh = OneHotEncoder(drop=[4], sparse=False, categories=[list(range(1,11))])
    dummies_train = oh.fit_transform(X_train[:,4].reshape(-1, 1))
    dummies_eval = oh.transform(X_eval[:,4].reshape(-1, 1))
    X_train = np.concatenate((X_train, dummies_train), axis=1)
    X_eval = np.concatenate((X_eval, dummies_eval), axis=1)
    X_5_kept_cats = [1,2,3,5,6,7,8,9,10]
    
    names = np.concatenate((names, ['X5_2_' + str(i) for i in X_5_kept_cats]))

    return X_train, y_train, X_eval, y_eval, names

def variable_sets():
    """
    Get sets of variables to use.
    """

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    # Sets_of features to try
    useful_num_vars = [43, 46, 53, 58, 60, 63]
    useful_zip_vars = [4, 17, 85, 86, 88, 89, 90, 91, 92]
    x_5 = list(range(93,102))

    pre_feature_sets = list(powerset([useful_num_vars, useful_zip_vars, x_5]))
    feature_sets_names = list(powerset(['useful_num_vars', 'useful_zip_vars', 'x_5']))

    feature_sets = []
    for _set in pre_feature_sets:
        compact_set = []
        for l in _set:
            compact_set += l
        feature_sets.append(compact_set)
    
    return feature_sets, feature_sets_names

def model_metrics(probs, y_real):
    most_prob = np.argsort(probs)[::-1][0:800]
    y_predict = np.zeros(len(y_real))
    y_predict[most_prob] = 1
    cm=confusion_matrix(y_real, y_predict)
    recall_1 =cm[1,1]/(cm[1,1]+cm[1,0])
    
    return recall_1

def dump_model(model_artifact, prefix_name):
        """
        Serializes and dumps model artifact
        """
        saving_path = os.path.realpath('./utilities/saved_models')
        filename =  prefix_name + '.pkl'
        file_path = os.path.join(saving_path, filename)
        pickle.dump(model_artifact, open(file_path, 'wb'))

        return file_path