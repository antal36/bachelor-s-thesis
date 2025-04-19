import pandas as pd
from sklearn.linear_model import LinearRegression
from .blendenpik_solver import BlendenpikSolver
from .qr_solver import QR_solver
from .iboss_solver import IBOSS_Solver
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Added r2_score import
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

nyse: pd.DataFrame = pd.read_csv(r"C:\Users\antal\Desktop\matfyz\bakalárka\stocks_data\prices.csv")
print(nyse.shape)

# Converting string dates into datetime type and extracting year
nyse["date"] = pd.to_datetime(nyse["date"], format="mixed").dt.year

nyse_encoded: pd.DataFrame = pd.get_dummies(nyse, columns=["symbol"], drop_first=True, dtype=int)

y = nyse_encoded["close"]
X = nyse_encoded.drop(columns=["close"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

#--------------------------------------------------------------------------------
mse_train_classic: list[float] = []
mse_test_classic: list[float] = []
time_classic: list[float] = []
r2_train_classic: list[float] = []   # Added R^2 lists for classic regression
r2_test_classic: list[float] = []

mse_train_iboss: list[float] = []
mse_test_iboss: list[float] = []
time_iboss: list[float] = []
r2_train_iboss: list[float] = []      # Added R^2 lists for IBOSS
r2_test_iboss: list[float] = []
    
mse_train_blendenpik: list[float] = []
mse_test_blendenpik: list[float] = []
time_blendenpik: list[float] = []
r2_train_blendenpik: list[float] = []   # Added R^2 lists for Blendenpik
r2_test_blendenpik: list[float] = []

#--------------------------------------------------------------------------------
for i in range(50):
    print(f"{i+1}/50")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    #--------------------------------------------------------------------------------
    # Classic regression
    lin_reg = LinearRegression()
    start_time = time.time()
    lin_reg.fit(X=X_train, y=y_train)
    end_time = time.time()
    time_classic.append(end_time - start_time)

    preds_train = lin_reg.predict(X=X_train)
    preds_test = lin_reg.predict(X=X_test)
    
    mse_train_classic.append(mean_squared_error(y_true=y_train, y_pred=preds_train))
    mse_test_classic.append(mean_squared_error(y_true=y_test, y_pred=preds_test))
    r2_train_classic.append(r2_score(y_true=y_train, y_pred=preds_train))
    r2_test_classic.append(r2_score(y_true=y_test, y_pred=preds_test))

    #--------------------------------------------------------------------------------
    # IBOSS
    iboss = IBOSS_Solver(X=X_train, y=y_train, r=60_000)
    start_time = time.time()
    iboss.solve_2()
    end_time = time.time()
    time_iboss.append(end_time - start_time)

    preds_train_iboss = iboss.predict(X=X_train)
    preds_test_iboss = iboss.predict(X=X_test)
    
    mse_train_iboss.append(mean_squared_error(y_true=y_train, y_pred=preds_train_iboss))
    mse_test_iboss.append(mean_squared_error(y_true=y_test, y_pred=preds_test_iboss))
    r2_train_iboss.append(r2_score(y_true=y_train, y_pred=preds_train_iboss))
    r2_test_iboss.append(r2_score(y_true=y_test, y_pred=preds_test_iboss))

    #--------------------------------------------------------------------------------
    # Blendenpik
    blenden = BlendenpikSolver(X=X_train, y=y_train, r=60_000)
    start_time = time.time()
    blenden.solve(method="unweighted", method_probs="shrinked")
    end_time = time.time()
    time_blendenpik.append(end_time - start_time)

    preds_train_blenden = blenden.predict(X=X_train)
    preds_test_blenden = blenden.predict(X=X_test)
    
    mse_train_blendenpik.append(mean_squared_error(y_true=y_train, y_pred=preds_train_blenden))
    mse_test_blendenpik.append(mean_squared_error(y_true=y_test, y_pred=preds_test_blenden))
    r2_train_blendenpik.append(r2_score(y_true=y_train, y_pred=preds_train_blenden))
    r2_test_blendenpik.append(r2_score(y_true=y_test, y_pred=preds_test_blenden))

#--------------------------------------------------------------------------------
print("Úplná regresia")
print("MSE_train: ", np.mean(mse_train_classic), "+-", np.std(mse_train_classic))
print("MSE_test: ", np.mean(mse_test_classic), "+-", np.std(mse_test_classic))
print("R²_train: ", np.mean(r2_train_classic), "+-", np.std(r2_train_classic))
print("R²_test: ", np.mean(r2_test_classic), "+-", np.std(r2_test_classic))
print("Čas: ",  np.mean(time_classic), "+-", np.std(time_classic))
print('---------------------------')

print("IBOSS")
print("MSE_train: ", np.mean(mse_train_iboss), "+-", np.std(mse_train_iboss))
print("MSE_test: ", np.mean(mse_test_iboss), "+-", np.std(mse_test_iboss))
print("R²_train: ", np.mean(r2_train_iboss), "+-", np.std(r2_train_iboss))
print("R²_test: ", np.mean(r2_test_iboss), "+-", np.std(r2_test_iboss))
print("Čas: ",  np.mean(time_iboss), '+-', np.std(time_iboss))
print('---------------------------')

print("Úplná Blendenpik")
print("MSE_train: ", np.mean(mse_train_blendenpik), "+-", np.std(mse_train_blendenpik))
print("MSE_test: ", np.mean(mse_test_blendenpik), '+-', np.std(mse_test_blendenpik))
print("R²_train: ", np.mean(r2_train_blendenpik), "+-", np.std(r2_train_blendenpik))
print("R²_test: ", np.mean(r2_test_blendenpik), "+-", np.std(r2_test_blendenpik))
print("Čas: ",  np.mean(time_blendenpik), '+-', np.std(time_blendenpik))
print('---------------------------')


"""Úplná regresia
MSE_train:  0.41269707839368763 +- 0.002811910483552525
MSE_test:  0.4188392451445519 +- 0.011346478966920794
R²_train:  0.9999410099492705 +- 4.129526404461713e-07
R²_test:  0.9999404587947056 +- 1.648137226502205e-06
Čas:  24.678476881980895 +- 0.9786232219175605
---------------------------
IBOSS
MSE_train:  0.4171870018742589 +- 0.0036583874095855247
MSE_test:  0.4237513501825094 +- 0.012077579413833224
R²_train:  0.9999403679987456 +- 5.507248626262577e-07
R²_test:  0.9999397615878551 +- 1.7125207282055918e-06
Čas:  18.65206676006317 +- 0.7312534168944236
---------------------------
Úplná Blendenpik
MSE_train:  0.4192330413237611 +- 0.004308136542264576
MSE_test:  0.4257236194195395 +- 0.011704214103086447
R²_train:  0.9999400759987173 +- 5.940835715674426e-07
R²_test:  0.9999394802129143 +- 1.6899134974560644e-06
Čas:  7.534196548461914 +- 0.5646257495396828
---------------------------"""