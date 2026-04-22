from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


def s_learner_discrete(train, test, X, T, y):
    train = train.copy()
    test = test.copy()

    model = LGBMRegressor()

    # Train on X + T
    model.fit(train[X + [T]], train[y])

    # Predict T=1 and T=0
    test_t1 = test.copy()
    test_t1[T] = 1

    test_t0 = test.copy()
    test_t0[T] = 0

    mu1 = model.predict(test_t1[X + [T]])
    mu0 = model.predict(test_t0[X + [T]])

    test["cate"] = mu1 - mu0
    return test


def t_learner_discrete(train, test, X, T, y):
    train = train.copy()
    test = test.copy()

    model_0 = LGBMRegressor()
    model_1 = LGBMRegressor()

    # Split
    train_0 = train[train[T] == 0]
    train_1 = train[train[T] == 1]

    model_0.fit(train_0[X], train_0[y])
    model_1.fit(train_1[X], train_1[y])

    mu0 = model_0.predict(test[X])
    mu1 = model_1.predict(test[X])

    test["cate"] = mu1 - mu0
    return test


def x_learner_discrete(train, test, X, T, y):
    train = train.copy()
    test = test.copy()

    # Step 1: outcome models
    model_0 = LGBMRegressor()
    model_1 = LGBMRegressor()

    train_0 = train[train[T] == 0]
    train_1 = train[train[T] == 1]

    model_0.fit(train_0[X], train_0[y])
    model_1.fit(train_1[X], train_1[y])

    # Step 2: pseudo effects
    tau_0 = model_1.predict(train_0[X]) - train_0[y]
    tau_1 = train_1[y] - model_0.predict(train_1[X])

    # Step 3: fit tau models
    tau_model_0 = LGBMRegressor()
    tau_model_1 = LGBMRegressor()

    tau_model_0.fit(train_0[X], tau_0)
    tau_model_1.fit(train_1[X], tau_1)

    # Step 4: propensity score
    prop = LogisticRegression(penalty=None)
    prop.fit(train[X], train[T])

    e = prop.predict_proba(test[X])[:, 1]

    tau0_pred = tau_model_0.predict(test[X])
    tau1_pred = tau_model_1.predict(test[X])

    test["cate"] = e * tau0_pred + (1 - e) * tau1_pred

    return test


def double_ml_cate(train, test, X, T, y):
    train = train.copy()
    test = test.copy()

    # Step 1: estimate E[T|X] and E[Y|X]
    model_t = LGBMRegressor()
    model_y = LGBMRegressor()

    model_t.fit(train[X], train[T])
    model_y.fit(train[X], train[y])

    T_hat = model_t.predict(train[X])
    Y_hat = model_y.predict(train[X])

    # Residuals
    T_res = train[T] - T_hat
    Y_res = train[y] - Y_hat

    # Avoid division by zero
    eps = 1e-6
    T_res = np.where(np.abs(T_res) < eps, eps, T_res)

    Y_star = Y_res / T_res
    w = T_res**2

    # Final model
    cate_model = LGBMRegressor()
    cate_model.fit(train[X], Y_star, sample_weight=w)

    test["cate"] = cate_model.predict(test[X])

    return test
