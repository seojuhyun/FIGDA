# -*- coding: utf-8 -*-
"""
This implementation is based on the original DevNet code from:
Guansong Pang, Chunhua Shen, and Anton van den Hengel.
Deep Anomaly Detection with Deviation Networks.
Proceedings of the 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2019).
https://doi.org/10.1145/3292500.3330871
Original implementation: Guansong Pang

Modifications and extensions:
Juhyun Seo (2026)
- TensorFlow 2.x compatibility updates
- Experimental pipeline integration for FIGDA
This code is provided for research and academic purposes.
"""

import numpy as np
import pandas as pd
import os, sys
import warnings
import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop


import matplotlib.pyplot as plt
import time
from scipy.sparse import vstack, csc_matrix
from devutils1 import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam



MAX_INT = np.iinfo(np.int32).max
data_format = 0


def dev_network_6d(input_shape):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(512, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(128, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(64, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl4')(intermediate)
    intermediate = Dense(32, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl5')(intermediate)
    intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_d(input_shape):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_s(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)    
    return Model(x_input, intermediate)

def dev_network_linear(input_shape):
    '''
    network architecture with no hidden layer, equivalent to linear mapping from
    raw inputs to anomaly scores
    '''    
    x_input = Input(shape=input_shape)
    intermediate = Dense(1, activation='linear',  name = 'score')(x_input)
    return Model(x_input, intermediate)

# Global variable for reference data
global_ref = K.constant(np.random.normal(loc=0.0, scale=1.0, size=5000), dtype='float32')


MEAN_REF = tf.constant(0.0, dtype=tf.float32)
STD_REF  = tf.constant(1.0, dtype=tf.float32)  # eps 포함

def deviation_loss(y_true, y_pred):
    confidence_margin = 5.0
    y_true = K.cast(y_true, dtype='float32')

    # z-score with fixed N(0,1)
    dev = (y_pred - MEAN_REF) / STD_REF

    inlier_loss  = tf.abs(dev)
    outlier_loss = tf.abs(tf.maximum(confidence_margin - dev, 0.))
    return tf.reduce_mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


def deviation_network(input_shape, network_depth, feature_importance_weights=None, alpha_dict=None):
    '''
    Construct the deviation network-based detection model.
    Supports hybrid initialization: alpha * FI + (1-alpha) * random
    Now supports per-layer alpha via alpha_dict: {layer_index: alpha_value}
    '''
    if network_depth == 6:
        model = dev_network_6d(input_shape)
    elif network_depth == 4:
        model = dev_network_d(input_shape)
    elif network_depth == 2:
        model = dev_network_s(input_shape)
    elif network_depth == 1:
        model = dev_network_linear(input_shape)
    else:
        sys.exit("❌ The network depth is not set properly")

    if feature_importance_weights is not None:
        fi = np.asarray(feature_importance_weights)
        
        
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                weights_shape = layer.get_weights()[0].shape
                input_dim = weights_shape[0]
                output_dim = weights_shape[1]

                # FI 준비
                fi_current = fi[:input_dim] if fi.shape[0] > input_dim else np.pad(fi, (0, max(0, input_dim - fi.shape[0])), 'constant')
                fi_weights = np.tile(fi_current.reshape(-1, 1), (1, output_dim))

                # Glorot-style 랜덤 초기화
                limit = np.sqrt(6 / (input_dim + output_dim))
                rand_weights = np.random.uniform(-limit, limit, size=(input_dim, output_dim))

                # alpha 적용
                alpha = alpha_dict.get(i, 0.0) if alpha_dict else 0.0
                blended_weights = alpha * fi_weights + (1 - alpha) * rand_weights
                bias = np.zeros(output_dim)

                layer.set_weights([blended_weights, bias])
                print(f"✅ Layer {i} initialized with FI (alpha={alpha}) - shape={blended_weights.shape}")



    rms = RMSprop(clipnorm=1.)
    model.compile(loss=deviation_loss, optimizer=rms)
    return model


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator"""
    #rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
    counter = 0
    while True:                
        ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        
        
        yield ref, training_labels
        counter += 1
        if counter > nb_batch:
            counter = 0

 

def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    dim = x_train.shape[1]   # 13?
    ref = np.empty((batch_size, dim), dtype=np.float32)  # Ensure inputs are float32...64, 13
    training_labels = []
    n_inliers = len(inlier_indices)   #전체 정상 데이터 개수
    n_outliers = len(outlier_indices)   #전체 비정상 데이터 개수
    for i in range(batch_size):    
        if i % 2 == 0:
            sid = rng.choice(n_inliers, 1)   # 무작위로 1개의 인덱스 선택
            ref[i] = x_train[inlier_indices[sid]]
            training_labels.append(0)
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels.append(1)
    return np.array(ref, dtype=np.float32), np.array(training_labels, dtype=np.float32)  # Ensure labels are float32


 
def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for libsvm stored sparse data.
    Alternates between positive and negative pairs.
    ''' 
    ref = np.empty((batch_size))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = x_train[ref, :].toarray()
    return ref, np.array(training_labels)


def load_model_weight_predict(model_name, input_shape, network_depth, x_test):
    '''
    load the saved weights to make predictions
    '''
    model = deviation_network(input_shape, network_depth)
    model.load_weights(model_name)
    scoring_network = Model(inputs=model.input, outputs=model.output)    
    
    if data_format == 0:
        scores = scoring_network.predict(x_test)
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores


def inject_noise_sparse(seed, n_out, random_seed):  
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    This is for sparse data.
    '''
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    print(noise.shape)
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()


def inject_noise(seed, n_out, random_seed):   
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''  
    random_seed=42
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise



from sklearn.model_selection import train_test_split

def prepare_data(data, test_size=0.2, val_size=0.2, random_state=42):
    """
    Train/Validation/Test set을 생성하는 함수입니다.

    Parameters:
    - data (DataFrame): 전체 데이터셋 (Time 컬럼 포함 가능)
    - test_size (float): 테스트셋 비율 (default=0.2)
    - val_size (float): 학습셋 내부에서 검증셋 비율 (default=0.2)
    - random_state (int): 랜덤 시드 고정

    Returns:
    - X_train, X_val, y_train, y_val, X_test, y_test, outlier_indices
    """

    # ✅ Feature, Label 분리
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # ✅ 전체에서 테스트셋 분리
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ✅ 학습셋에서 검증셋 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )

    # ✅ 이상치 인덱스 추출 (학습 + 검증셋 기준)
    outlier_indices = np.where(y_train_val == 1)[0]

    # ✅ 디버깅 정보 출력
    print(f"🔹 Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    print(f"🔹 Outliers in Train+Val: {len(outlier_indices)}")

    return X_train, X_val, y_train, y_val, X_test, y_test, outlier_indices


import csv

def save_results_extended(dataset_name, script_name, auc_roc, auc_pr, accuracy, precision, recall, f1, output_file="results_detailed.csv"):
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Dataset", "Script", "AUC-ROC", "AUC-PR", "Accuracy", "Precision", "Recall", "F1-Score"])
        writer.writerow([dataset_name, script_name, auc_roc, auc_pr, accuracy, precision, recall, f1])

    print(f"✅ Detailed results saved to {output_file}")