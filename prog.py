import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataLocation = ""  # input location of data file here (eg, "C:/Destop/data.xlsx")

df = pd.read_excel(dataLocation)

df.fillna(0, inplace = True)

#modifying the dataframe
start = 2
end = 531

Location = df.iloc[start : end, 0]

pH = df.iloc[start : end, 1].astype(np.float64)
EC = df.iloc[start : end, 2].astype(np.float64)
TDS = df.iloc[start : end, 3].astype(np.float64)
ORP = df.iloc[start : end, 4].astype(np.float64)
Ca = df.iloc[start : end, 5].astype(np.float64)
Mg = df.iloc[start : end, 6].astype(np.float64)
Na = df.iloc[start : end, 7].astype(np.float64)
K = df.iloc[start : end, 8].astype(np.float64)
HCO3 = df.iloc[start : end, 9].astype(np.float64)
SO4 = df.iloc[start : end, 10].astype(np.float64)
Cl = df.iloc[start : end, 11].astype(np.float64)
NO3 = df.iloc[start : end, 12].astype(np.float64)
CO3 = df.iloc[start : end, 13].astype(np.float64)
F = df.iloc[start : end, 14].astype(np.float64)
Cu = df.iloc[start : end, 15].astype(np.float64)
Pb = df.iloc[start : end, 16].astype(np.float64)
Zn = df.iloc[start : end, 17].astype(np.float64)
U = df.iloc[start : end, 18].astype(np.float64)
NH4 = df.iloc[start : end, 19].astype(np.float64)
Br = df.iloc[start : end, 20].astype(np.float64)

date = df.iloc[start : end, 21]

df = pd.concat([Location, pH, EC, TDS, ORP, Ca, Mg, Na, K, HCO3, SO4, Cl, NO3, CO3, F, Cu, Pb, Zn, U, NH4, Br, date], axis = 1)
df.columns = ['Location', 'pH', 'EC', 'TDS', 'ORP', 'Ca', 'Mg', 'Na', 'K', 'HCO3', 'SO4', 'Cl', 'NO3', 'CO3', 'F', 'Cu', 'Pb', 'Zn', 'U', 'NH4', 'Br', 'Date']

#calculation of pH
df['npH'] = df.pH.apply(lambda x: (100 if (8.5 >= x >= 7)  
                                 else(80 if  (8.6 >= x >= 8.5) or (6.9 >= x >= 6.8) 
                                      else(60 if (8.8 >= x >= 8.6) or (6.8 >= x >= 6.7) 
                                          else(40 if (9 >= x >= 8.8) or (6.7 >= x >= 6.5)
                                              else 0)))))

#calculation of EC
df['nEC'] = df.EC.apply(lambda x: (100 if (1900 >= x >= 1800)  
                                 else(80 if  (2200 >= x > 1900) or (1800 > x >= 1500) 
                                      else(60 if (2500 >= x > 2200) or (1500 > x >= 1200) 
                                          else(40 if (2800 >= x > 2500) or (1200 > x >= 900)
                                              else 0)))))

#calculation of TDS
df['nTDS'] = df.TDS.apply(lambda x: (100 if (1500 >= x >= 500)  
                                 else(80 if  (2000 >= x > 1500) or (500 > x >= 400) 
                                      else(60 if (3000 >= x > 2500) or (400 > x >= 300) 
                                          else(40 if (3500 >= x > 3000) or (300 > x >= 200)
                                              else 0)))))

#calculation of ORP
df['nORP'] = df.ORP.apply(lambda x: (100 if (60 >= x >= 20)  
                                 else(80 if  (65 >= x >= 60) or (20 >= x >= 15)
                                      else(60 if (70 >= x >= 65) or (15 >= x >= 10) 
                                          else(40 if (75 >= x >= 70) or (10 >= x >= 5)
                                              else 0)))))

#calculation of Ca
df['nCa'] = df.Ca.apply(lambda x: (100 if (75 >= x >= 70)  
                                 else(80 if  (80 >= x >= 75) or (70 >= x >= 65) 
                                      else(60 if (85 >= x >= 80) or (65 >= x >= 60) 
                                          else(40 if (90 >= x >= 85) or (60 >= x >= 55)
                                              else 0)))))

#calculation of Mg
df['nMg'] = df.Mg.apply(lambda x: (100 if (55 >= x)  
                                 else(80 if  (60 >= x) 
                                      else(60 if (65 >= x)
                                          else(40 if (70 >= x)
                                              else 0)))))

#calculation of Na
df['nNa'] = df.Na.apply(lambda x: (100 if (210 >= x)  
                                 else(80 if  (220 >= x)
                                      else(60 if (230 >= x)
                                          else(40 if (240 >= x)
                                              else 0)))))

#calculation of K
df['nK'] = df.K.apply(lambda x: (100 if (5.5 >= x >= 5.0)
                                 else(80 if  (6.0 >= x)
                                      else(60 if (6.5 >= x)
                                          else(40 if (7.0 >= x)
                                              else 0)))))

#calculation of HCO3
df['nHCO3'] = df.HCO3.apply(lambda x: (100 if (1000 >= x)
                                 else(80 if  (1100 >= x) 
                                      else(60 if (1200 >= x) 
                                          else(40 if (1300 >= x)
                                              else 0)))))

#calculation of SO4
df['nSO4'] = df.SO4.apply(lambda x: (100 if (400 >= x)
                                 else(80 if  (500 >= x) 
                                      else(60 if (600 >= x) 
                                          else(40 if (700 >= x)
                                              else 0)))))

#calculation of Cl
df['nCl'] = df.Cl.apply(lambda x: (100 if (400 >= x)
                                 else(70 if  (450 >= x) 
                                      else(40 if (500 >= x) 
                                          else(10 if (550 >= x)
                                              else 0)))))

#calculation of NO3
df['nNO3'] = df.NO3.apply(lambda x: (100 if (30 >= x)
                                 else(70 if  (40 >= x) 
                                      else(0))))

#calculation of CO3
df['nCO3'] = df.CO3.apply(lambda x: (100 if (4 >= x >= 3)  
                                 else(80 if  (5 >= x >= 4) or (3.0 >= x >= 2.5) 
                                      else(60 if (6 >= x >= 5) or (2.5 >= x >= 2) 
                                          else(40 if (7 >= x >= 6) or (2 >= x >= 1)
                                              else 0)))))

#calculation of F
df['nF'] = df.F.apply(lambda x: (100 if (0.4 >= x >= 0.3)  
                                 else(80 if  (0.6 >= x) 
                                      else(60 if (1 >= x) 
                                          else(40 if (1.4 >= x)
                                              else 0)))))

#calculation of Cu
df['nCu'] = df.Cu.apply(lambda x: (100 if (5 >= x)  
                                 else(80 if  (6 >= x) 
                                      else(60 if (7 >= x) 
                                          else(40 if (10 >= x)
                                              else 0)))))

#calculation of Pb
df['nPb'] = df.Pb.apply(lambda x: (100 if (5 >= x)  
                                 else(80 if  (6 >= x) 
                                      else(60 if (7 >= x) 
                                          else(40 if (10 >= x)
                                              else 0)))))

#calculation of Zn
df['nZn'] = df.Zn.apply(lambda x: (100 if (50 >= x >= 40)  
                                 else(80 if  (60 >= x >= 50) or (40 >= x >= 30) 
                                      else(60 if (70 >= x >= 60) or (30 >= x >= 20) 
                                          else(40 if (80 >= x >= 70) or (20 >= x >= 10)
                                              else 0)))))

#calculation of U
df['nU'] = df.U.apply(lambda x: (100 if (20 >= x)  
                                 else(80 if  (30 >= x) 
                                      else(60 if (35 >= x) 
                                          else(40 if (40 >= x)
                                              else 0)))))

#calculation of NH4
df['nNH4'] = df.NH4.apply(lambda x: (100 if (0.5 >= x)  
                                 else(80 if  (1 >= x) 
                                      else(60 if (1.5 >= x) 
                                          else(40 if (2 >= x)
                                              else 0)))))

#calculation of Br
df['nBr'] = df.Br.apply(lambda x: (100 if (2 >= x)  
                                 else(80 if  (2.5 >= x) 
                                      else(60 if (3 >= x) 
                                          else(40 if (3.5 >= x)
                                              else 0)))))

#weighted normalization
df['wpH'] = df.npH * 0.08
df['wEC'] = df.nEC * 0.004
df['wTDS'] = df.nTDS * 0.053
df['wORP'] = df.nORP * 0.053

df['wCa'] = df.nCa * 0.053
df['wMg'] = df.nMg * 0.053
df['wNa'] = df.nNa * 0.014
df['wK'] = df.nK * 0.053
df['wHCO3'] = df.nHCO3 * 0.053
df['wSO4'] = df.nSO4 * 0.053
df['wCl'] = df.nCl * 0.053
df['wNO3'] = df.nNO3 * 0.053
df['wCO3'] = df.nCO3 * 0.053
df['wF'] = df.nF * 0.053
df['wCu'] = df.nCu * 0.053
df['wPb'] = df.nPb * 0.053
df['wZn'] = df.nZn * 0.053
df['wU'] = df.nU * 0.053
df['wNH4'] = df.nNH4 * 0.053
df['wBr'] = df.nBr * 0.053

df['wqi'] = df.wpH + df.wEC + df.wTDS + df.wORP + df.wCa + df.wMg + df.wNa + df.wK + df.wHCO3 + df.wSO4 + df.wCl + df.wNO3 + df.wCO3 + df.wF + df.wCu + df.wPb + df.wZn + df.wU + df.wNH4 + df.wBr

#convert datetime value to int64 for ML model
df['Date'] = df['Date'].values.astype('int64')

#aggregation
aggr = df.groupby('Date')['wqi'].mean()

#removing misaligned values
data = aggr.reset_index(level = 0, inplace = False)
data = data[np.isfinite(data['wqi'])]

#using linear regression for prediction
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#resetting levels
data = aggr.reset_index(level = 0, inplace = False)
data = data[np.isfinite(data['wqi'])]

cols = ['Date']

x = data[cols]
y = data['wqi']

reg = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

reg.fit(x_train, y_train)

a = reg.predict(x_test)

from sklearn.metrics import mean_squared_error
init_mse = mean_squared_error(y_test, a)

#using gradient descent to optimize it further
x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x]

alpha = 0.1
iterations = 3000
m = y.size
np.random.seed(4)
theta = np.random.rand(2)

def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_costs

past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)
theta = past_thetas[-1]

newBatch = [theta[0], theta[1]]

def rmse(y, y_pred):
    rmse = np.sqrt(sum(y - y_pred))
    return rmse
   
y_pred = x.dot(newBatch)

dt = pd.DataFrame({'Actual': y, 'Predicted': y_pred})  
dt = pd.concat([data, dt], axis = 1)

#model accuracy
from sklearn import metrics
from sklearn.metrics import r2_score
acc = r2_score(y, y_pred) * 100
final_mse = np.sqrt(metrics.mean_squared_error(y, y_pred))

#display stats
print("\n\nStats :-")
print("1) Mean squared error : {:.2f}".format(init_mse))
print("2) Accuracy : {:.2f} %".format(acc))
print("3) Gradient Descent : {:.2f}, {:.2f}".format(theta[0], theta[1]))

#plotting predicted vs actual value
dt['Date'] = pd.to_datetime(dt['Date'])
x_axis = dt.Date
y_axis = dt.Actual
y1_axis = dt.Predicted

plt.scatter(x_axis, y_axis)
plt.plot(x_axis, y1_axis, color = 'b')
plt.title("linear regression")
plt.xlabel("Date")
plt.ylabel("WQI")

plt.show()
