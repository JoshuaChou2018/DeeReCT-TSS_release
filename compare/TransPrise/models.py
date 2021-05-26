"""
TransPrise model for learning and prediction
"""

import keras.layers as l
import keras.activations as a
from keras import backend
from keras.engine.topology import Container
from keras.models import Sequential, Model


def classification_model(data_shape):
    """
    
    :param data_shape: input data shape (length and layers)
    :return: models for classification
    """
    inp = l.Input(data_shape)
    
    conv_1 = l.Conv1D(128, 2, padding='same', activation='relu', use_bias=False)(inp)
    conv_2 = l.Conv1D(128, 4, padding='same', activation='relu', use_bias=False)(inp)
    conv_3 = l.Conv1D(128, 8, padding='same', activation='relu', use_bias=False)(inp)
    conv_4 = l.Conv1D(128, 16, padding='same', activation='relu', use_bias=False)(inp)
    
    concat = l.concatenate([conv_1, conv_2, conv_3, conv_4], axis=-1)
                                                    
    conv_concat_1 = l.Conv1D(128, 1, padding='same', activation='relu', use_bias=False)(concat)
    batch_norm_1 = l.BatchNormalization(axis=1)(conv_concat_1)
    batch_pool_1 = l.MaxPooling1D(pool_size=2)(batch_norm_1)
    conv_concat_2 = l.Conv1D(16, 1, padding='same', activation='relu', use_bias=False)(batch_pool_1)
    batch_norm_2 = l.BatchNormalization(axis=1)(conv_concat_2)
    batch_pool_2 = l.MaxPooling1D(pool_size=2)(batch_norm_2)
    drop = l.Dropout(0.5)(batch_pool_2)
    flat = l.Flatten()(drop)
    
    dense_1 = l.Dense(256, activation='relu', use_bias=False)(flat)
    dense_2 = l.Dense(128, activation='relu', use_bias=False)(dense_1)
    batch_norm_2 = l.BatchNormalization(axis=1)(dense_2)
    out = l.Dense(1, activation='sigmoid')(batch_norm_2)

    class_mod = Model(input=inp, output=out)
    class_mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return class_mod


def regression_model(data_shape):
    """
    
    :param data_shape: input data shape (length and layers)
    :return: models for classification
    """
    inp = l.Input(data_shape)
    
    conv_1 = l.Conv1D(128, 2, padding='same', activation='relu', use_bias=False)(inp)
    conv_2 = l.Conv1D(128, 4, padding='same', activation='relu', use_bias=False)(inp)
    conv_3 = l.Conv1D(128, 8, padding='same', activation='relu', use_bias=False)(inp)
    conv_4 = l.Conv1D(128, 16, padding='same', activation='relu', use_bias=False)(inp)
    
    concat = l.concatenate([conv_1, conv_2, conv_3, conv_4], axis=-1)
                                                    
    conv_concat_1 = l.Conv1D(128, 1, padding='same', activation='relu', use_bias=False)(concat)
    batch_norm_1 = l.BatchNormalization(axis=1)(conv_concat_1)
    batch_pool_1 = l.MaxPooling1D(pool_size=2)(batch_norm_1)
    conv_concat_2 = l.Conv1D(16, 1, padding='same', activation='relu', use_bias=False)(batch_pool_1)
    batch_norm_2 = l.BatchNormalization(axis=1)(conv_concat_2)
    batch_pool_2 = l.MaxPooling1D(pool_size=2)(batch_norm_2)
    drop = l.Dropout(0.5)(batch_pool_2)
    flat = l.Flatten()(drop)
    
    dense_1 = l.Dense(256, activation='relu', use_bias=False)(flat)
    dense_2 = l.Dense(128, activation='relu', use_bias=False)(dense_1)
    batch_norm_2 = l.BatchNormalization(axis=1)(dense_2)
    out = l.Dense(1, activation='linear')(batch_norm_2)

    regr_mod = Model(input=inp, output=out)
    regr_mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    
    return regr_mod


def reset_weights(model):
    """
    
    :param model: models for reset weights
    :return: models with initialized weights
    """

    session = backend.get_session()
    for layer in model.layers:
        if isinstance(layer, Container):
            reset_weights(layer)
            continue
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)


def cv_split(examples, folds):
    """
    
    :param examples: how many examples in set
    :param folds: how many folds you want
    :return: iterator that yield train and val split
    """
    examples_in_fold = examples // folds
    for i in range(folds):
        val = [j for j in range(examples_in_fold*i, examples_in_fold*(i+1))]
        train = [j for j in range(0, examples_in_fold*i)] + [j for j in range(examples_in_fold*i, examples)]
        yield train, val
