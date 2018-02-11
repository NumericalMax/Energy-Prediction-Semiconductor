RANDOM_STATE = 1337

import pandas as pd
import numpy as np
np.random.seed(RANDOM_STATE)

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull

def rmsle(y_pred, y_target):
    """Computation of the root mean squared logarithmic error.
    
    Requirements:
        numpy
    
    Args:
        y_pred (numpy float array): The predicted values
        y_target (numpy float array): The target values
    
    Remark:
        The input arrays have to be of same length
    
    Returns:
        float: Root mean squared logarithmic error of the prediction.

    """
    assert len(y_pred) == len(y_target)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_target), 2)))
    
def save_submission(prediction_1, prediction_2, test_id):
    """The function saves the prediction values to a csv file
    
    Requirements:
        pandas
    
    Args:
        prediction_1 (numpy float array): The predicted values for the first target: formation_energy_ev_natom
        prediction_2 (numpy float array): The predicted values for the first target: bandgap_energy_ev
        test_id (numpy int array): The ids of the test data
    """

    submission = pd.concat([test_id, pd.DataFrame(prediction_1), pd.DataFrame(prediction_2)], axis=1)
    submission.columns = ['id','formation_energy_ev_natom', 'bandgap_energy_ev']
    submission.to_csv('submission.csv', index = False)
    
def get_xyz_data(filename, ids):
    """The function loads the xyz-geometry files and transforms it into a common python format (pandas table)
    
    Remark:
        The parts for read and split are adopted from Tony Y: https://www.kaggle.com/tonyyy
    
    TODO:
        I am pretty sure that there is a faster and more elegant solution in order to generate the pandas dataframe
    
    Requirements:
        pandas
    
    Args:
        filename (string): path of the xyz file
        ids (integer): id of the corresponding entry in train data
        
    Returns:
        pandas dataframe A: Geometry data from the xyz file in table format
        pandas dataframe B: Lattice data from the xyz file in table format
        
    """
    
    A = pd.DataFrame(columns=list('ABCDE'))
    B = pd.DataFrame(columns=list('ABCE'))
    
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':

                newrowA = pd.DataFrame([[x[1],x[2],x[3],x[4],ids]], columns=list('ABCDE'))
                A = A.append(newrowA)
                
            elif x[0] == 'lattice_vector':
                
                newrowB = pd.DataFrame([[x[1],x[2],x[3],ids]], columns=list('ABCE'))
                B = B.append(newrowB)

    return A, B
    
def one_hot(df):
    """The function performs one hot encoding on the spacegroup column, which is not ordinal.

    Requirements:
        pandas
    
    Args:
        df (pandas dataframe): Data table with spacegroup column
    
    Returns:
        pandas dataframe df: Dataframe with one hot encoded spacegroup column
        
    """
    
    s = pd.Series(df["spacegroup"])
    t = pd.get_dummies(s)
    
    df["spacegroup_12"] = t[12]
    df["spacegroup_33"] = t[33]
    df["spacegroup_167"] = t[167]
    df["spacegroup_194"] = t[194]
    df["spacegroup_206"] = t[206]
    df["spacegroup_227"] = t[227]
    df = df.drop("spacegroup", axis = 1)
    
    return df
    
def feature_extraction(df, df1, n):
 
    """The function performs feature extraction on the information from xyz files
    
    Requirements:
        numpy
    
    Args:
        df (pandas dataframe): Data table with spatial distribution and atom info
        df1 (pandas dataframe): Data table with lattice vectors (obsolet in this version)
        n (int): length of feature matrix for preallocation
        
    Returns:
        pandas dataframe feat_matrix: Table with features which are extracted from the input data
        
    """

    feat_matrix = pd.DataFrame(index = range(1,n+1), columns=["id"])
    
    # features we like to extract
    # just exemplary: time to become creative here
    variance_x = np.zeros(n, dtype=float)
    variance_y = np.zeros(n, dtype=float)
    variance_z = np.zeros(n, dtype=float)
    
    variance_x_ga = np.zeros(n, dtype=float)
    variance_y_ga = np.zeros(n, dtype=float)
    variance_z_ga = np.zeros(n, dtype=float)
    
    variance_x_al = np.zeros(n, dtype=float)
    variance_y_al = np.zeros(n, dtype=float)
    variance_z_al = np.zeros(n, dtype=float)
    
    variance_x_o = np.zeros(n, dtype=float)
    variance_y_o = np.zeros(n, dtype=float)
    variance_z_o = np.zeros(n, dtype=float)
    
    variance_x_in = np.zeros(n, dtype=float)
    variance_y_in = np.zeros(n, dtype=float)
    variance_z_in = np.zeros(n, dtype=float)
    
    on_same_axis_x = np.zeros(n, dtype=float)
    on_same_axis_y = np.zeros(n, dtype=float)
    on_same_axis_z = np.zeros(n, dtype=float)
    
    for index in range(n):
        
        #ALL
        matrix = df[df["E"]==index+1]
        matrix = matrix[["A","B","C"]].as_matrix()
        matrix = matrix.astype(float)
        
        variance_x[index] = np.var(matrix[:,0])
        variance_y[index] = np.var(matrix[:,1])
        variance_z[index] = np.var(matrix[:,2])

        pca = PCA(n_components=3)
        X_r = pca.fit(matrix).transform(matrix)
        df_ = pd.DataFrame(np.round(X_r,3))
        on_same_axis_x[index] = len(df_[0].unique())
        on_same_axis_y[index] = len(df_[1].unique())
        on_same_axis_z[index] = len(df_[2].unique())
        
        
        #GA
        matrix = df[df["E"]==index+1]
        matrix = matrix[matrix["D"]=='Ga']
        matrix = matrix[["A","B","C"]].as_matrix()
        matrix = matrix.astype(float)
        if(len(matrix) > 0):
            variance_x_ga[index] = np.var(matrix[:,0])
            variance_y_ga[index] = np.var(matrix[:,1])
            variance_z_ga[index] = np.var(matrix[:,2])
        
        #AL
        matrix = df[df["E"]==index+1]
        matrix = matrix[matrix["D"]=='Al']
        matrix = matrix[["A","B","C"]].as_matrix()
        matrix = matrix.astype(float)
        if(len(matrix) > 0):
            variance_x_al[index] = np.var(matrix[:,0])
            variance_y_al[index] = np.var(matrix[:,1])
            variance_z_al[index] = np.var(matrix[:,2])
        
        #O
        matrix = df[df["E"]==index+1]
        matrix = matrix[matrix["D"]=='O']
        matrix = matrix[["A","B","C"]].as_matrix()
        matrix = matrix.astype(float)
        if(len(matrix) > 0):
            variance_x_o[index] = np.var(matrix[:,0])
            variance_y_o[index] = np.var(matrix[:,1])
            variance_z_o[index] = np.var(matrix[:,2])
        
        #IN
        matrix = df[df["E"]==index+1]
        matrix = matrix[matrix["D"]=='In']
        matrix = matrix[["A","B","C"]].as_matrix()
        matrix = matrix.astype(float)
        if(len(matrix) > 0):
            variance_x_in[index] = np.var(matrix[:,0])
            variance_y_in[index] = np.var(matrix[:,1])
            variance_z_in[index] = np.var(matrix[:,2])

    feat_matrix["variance_x"] = variance_x
    feat_matrix["variance_y"] = variance_y
    feat_matrix["variance_z"] = variance_z
    
    feat_matrix["variance_x_ga"] = variance_x_ga
    feat_matrix["variance_y_ga"] = variance_y_ga
    feat_matrix["variance_z_ga"] = variance_z_ga
    
    feat_matrix["variance_x_al"] = variance_x_al
    feat_matrix["variance_y_al"] = variance_y_al
    feat_matrix["variance_z_al"] = variance_z_al
    
    feat_matrix["variance_x_o"] = variance_x_o
    feat_matrix["variance_y_o"] = variance_y_o
    feat_matrix["variance_z_o"] = variance_z_o
    
    feat_matrix["variance_x_in"] = variance_x_in
    feat_matrix["variance_y_in"] = variance_y_in
    feat_matrix["variance_z_in"] = variance_z_in
    
    feat_matrix["on_same_axis_x"] = on_same_axis_x
    feat_matrix["on_same_axis_y"] = on_same_axis_y
    feat_matrix["on_same_axis_z"] = on_same_axis_z
    
    return feat_matrix

 
def merge_df(feat_atom_mat, feat_mat):
    
    """The function performs a merge on two pandas tables
    
    Requirements:
        pandas
    
    Args:
        feat_atom_mat (pandas dataframe): First feature matrix
        feat_mat (pandas dataframe): Second feature matrix
        
    Remark:
        The ids have to match entry wise
    
    Returns:
        pandas dataframe full: Merged feature matrix
        
    """
    
    feat_atom_mat = feat_atom_mat.fillna(0)
    full = pd.concat([feat_mat, feat_atom_mat], axis=1, join_axes=[feat_mat.index])
    full = full.drop("id", axis = 1)
    
    return full
    
if __name__ == "__main__":
    
    # load and prepare data
    print("Load data")
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    train_id = train["id"]
    test_id = test["id"]
    label = train[["formation_energy_ev_natom", "bandgap_energy_ev"]]
    n_train = len(train)
    n_test = len(test)
    
    # load xyz-files
    print("Load geomatry (.xyz) files - takes approx. 6 minutes")
    train_atoms = pd.DataFrame(columns=list('ABCDE'))
    train_lattices = pd.DataFrame(columns=list('ABCE'))
    for k in range(n_train):

        idx = train.id.values[k]
        fn = "../input/train/{}/geometry.xyz".format(idx)
        train_xyz, train_lat = get_xyz_data(fn, k+1)
        train_atoms = train_atoms.append(train_xyz)
        train_lattices = train_lattices.append(train_lat)
    
    test_atoms = pd.DataFrame(columns=list('ABCDE'))
    test_lattices = pd.DataFrame(columns=list('ABCE'))
    for k in range(n_test):

        idx = test.id.values[k]
        fn = "../input/test/{}/geometry.xyz".format(idx)
        test_xyz, test_lat = get_xyz_data(fn, k+1)
        test_atoms = test_atoms.append(test_xyz)
        test_lattices = test_lattices.append(test_lat)
    
    train = train.drop(["id", "formation_energy_ev_natom", "bandgap_energy_ev"], axis = 1)
    test = test.drop("id", axis = 1)
    
    # one hot encode of spacegroup
    print("One hot encode of 'Spacegroup' column")
    train = one_hot(train)
    test = one_hot(test)
    
    # feature engineering with ase library
    print("Perform feature extraction on .xyz files - takes approx. 6 minutes")
    feat_matrix_train = feature_extraction(train_atoms, train_lattices, n_train)
    feat_matrix_test = feature_extraction(test_atoms, test_lattices, n_test)
    
    # merge data
    print("Merge data")
    train_full = merge_df(feat_matrix_train, train)
    test_full = merge_df(feat_matrix_test, test)
    
    print(np.max(train_full))
    
    # train - val split
    print("Train and validation split")
    train_full = train_full.fillna(0)
    
    X_train, X_val, y_train, y_val = train_test_split(train_full, label, test_size=0.15, random_state=RANDOM_STATE)
    
    # define models
    alg_1_a = XGBRegressor(learning_rate = 0.1, n_estimators=230, max_depth=3,
                           min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                           objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=RANDOM_STATE)
    alg_2_a = XGBRegressor(learning_rate = 0.1, n_estimators=230, max_depth=3,
                            min_child_weight=0, gamma=0, subsample=0.8,
                            colsample_bytree=0.8, objective= 'reg:linear',
                            nthread=4, scale_pos_weight=1, seed=RANDOM_STATE)
                           
    param_1_b = {'num_leaves': 4, 'objective': 'regression', 'min_data_in_leaf': 15, 'learning_rate': 0.01,'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'metric': 'l2', 'num_threads': 4}
    param_2_b = {'num_leaves': 4, 'objective': 'regression', 'min_data_in_leaf': 15, 'learning_rate': 0.01,'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'metric': 'l2', 'num_threads': 4}

    MAX_ROUNDS = 10000
    
    param_1_c = {'max_depth': 4, 'eta': 0.01, 'silent': 1, 'objective': 'reg:linear', 'eval_metric': 'rmse'}
    param_2_c = {'max_depth': 4, 'eta': 0.01, 'silent': 1, 'objective': 'reg:linear', 'eval_metric': 'rmse'}
    
    # train models and evaluate error
    print("Train XGBRegressor Model for formation_energy_ev_natom")
    alg_1_a.fit(X_train, y_train["formation_energy_ev_natom"])
    pred_val_1_a = alg_1_a.predict(X_val)
    pred_test_1_a = alg_1_a.predict(test_full)
    pred_train_1_a = alg_1_a.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_1_a, y_val["formation_energy_ev_natom"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_1_a, label["formation_energy_ev_natom"])))
    
    print("Train Light Gradient Boosting Model for formation_energy_ev_natom")
    dtrain = lgb.Dataset(X_train, label=y_train["formation_energy_ev_natom"])
    alg_1_b = lgb.train(param_1_b, dtrain, num_boost_round=MAX_ROUNDS,valid_sets=[dtrain], early_stopping_rounds=1000, verbose_eval=5000)
    pred_val_1_b = alg_1_b.predict(X_val)
    pred_test_1_b = alg_1_b.predict(test_full)
    pred_train_1_b = alg_1_b.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_1_b, y_val["formation_energy_ev_natom"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_1_b, label["formation_energy_ev_natom"])))
    
    print("Train xgboost Model for formation_energy_ev_natom")
    alg_1_c = xgb.train(param_1_c, xgb.DMatrix(X_train, label=y_train["formation_energy_ev_natom"]), num_boost_round = 2000)
    pred_val_1_c = alg_1_c.predict(xgb.DMatrix(X_val))
    pred_test_1_c = alg_1_c.predict(xgb.DMatrix(test_full))
    pred_train_1_c = alg_1_c.predict(xgb.DMatrix(train_full))
    print("RMSLE for validation data: " + str(rmsle(pred_val_1_c, y_val["formation_energy_ev_natom"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_1_c, label["formation_energy_ev_natom"])))

    print("Train XGBRegressor Model for bandgap_energy_ev")
    alg_2_a.fit(X_train, y_train["bandgap_energy_ev"])
    pred_val_2_a = alg_2_a.predict(X_val)
    pred_test_2_a = alg_2_a.predict(test_full)
    pred_train_2_a = alg_2_a.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_2_a, y_val["bandgap_energy_ev"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_2_a, label["bandgap_energy_ev"])))
    
    print("Train Light Gradient Boosting Model for bandgap_energy_ev")
    dtrain = lgb.Dataset(X_train, label=y_train["bandgap_energy_ev"])
    alg_2_b = lgb.train(param_2_b, dtrain, num_boost_round=MAX_ROUNDS,valid_sets=[dtrain], early_stopping_rounds=1000, verbose_eval=5000)
    pred_val_2_b = alg_2_b.predict(X_val)
    pred_test_2_b = alg_2_b.predict(test_full)
    pred_train_2_b = alg_2_b.predict(train_full)
    print("RMSLE for validation data: " + str(rmsle(pred_val_1_b, y_val["bandgap_energy_ev"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_1_b, label["bandgap_energy_ev"])))
    
    print("Train xgboost Model for bandgap_energy_ev")
    alg_2_c = xgb.train(param_2_c, xgb.DMatrix(X_train, label=y_train["bandgap_energy_ev"]), num_boost_round = 2000)    
    pred_val_2_c = alg_2_c.predict(xgb.DMatrix(X_val))
    pred_test_2_c = alg_2_c.predict(xgb.DMatrix(test_full))
    pred_train_2_c = alg_2_c.predict(xgb.DMatrix(train_full))
    print("RMSLE for validation data: " + str(rmsle(pred_val_2_c, y_val["bandgap_energy_ev"])))
    print("RMSLE for complete train data: " + str(rmsle(pred_train_2_c, label["bandgap_energy_ev"])))
    
    # ensemble
    res_1 = [1./3., 1./3., 1./3.]
    res_2 = [1./3., 1./3., 1./3.]
    pred_test_1 = res_1[0]*pred_test_1_a + res_1[1]*pred_test_1_b + res_1[2]*pred_test_1_c
    pred_val_1 = res_1[0]*pred_val_1_a + res_1[1]*pred_val_1_b + res_1[2]*pred_val_1_c
    pred_train_1 = res_1[0]*pred_train_1_a + res_1[1]*pred_train_1_b + res_1[2]*pred_train_1_c
    val_error_1 = rmsle(pred_val_1, y_val["formation_energy_ev_natom"])
    train_error_1 = rmsle(pred_train_1, label["formation_energy_ev_natom"])
    print("RMSLE for validation data on formation_energy_ev_natom after ensemble: " + str(val_error_1))
    print("RMSLE for complete train data on formation_energy_ev_natom after ensemble: " + str(train_error_1))
    
    pred_test_2 = res_2[0]*pred_test_2_a + res_2[1]*pred_test_2_b + res_2[2]*pred_test_2_c
    pred_val_2 = res_2[0]*pred_val_2_a + res_2[1]*pred_val_2_b + res_2[2]*pred_val_2_c
    pred_train_2 = res_2[0]*pred_train_2_a + res_2[1]*pred_train_2_b + res_2[2]*pred_train_2_c
    val_error_2 = rmsle(pred_val_2, y_val["bandgap_energy_ev"])
    train_error_2 = rmsle(pred_train_2, label["bandgap_energy_ev"])
    print("RMSLE for validation data on bandgap_energy_ev after ensemble: " + str(val_error_2))
    print("RMSLE for complete train data on bandgap_energy_ev after ensemble: " + str(train_error_2))
    
    print("Total validation error (formation_energy_ev_natom and bandgap_energy_ev): " + str((val_error_1 + val_error_2)/2.0))
    print("Total train error: (formation_energy_ev_natom and bandgap_energy_ev): " + str((train_error_1 + train_error_2)/2.0))
    
    pred_test_1[pred_test_1<=0] = 0.0001
    pred_test_2[pred_test_2<=0] = 0.0001
    
    # save submission
    print("Save submission")
    save_submission(pred_test_1, pred_test_2, test_id)
    
    print("Finished")