""" Create a OLS model """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from scripts.msa import MSA_collection
from scripts.helpers import DATA_PATH

data_path = DATA_PATH / "fasta"

# create MSA
msa = MSA_collection(data_path)

# extract time from filenames
t = [int(name.stem.split('_')[-1][1:]) for name in list(data_path.glob('*.fasta'))]
t.sort()

# calculate diversities
div = np.array([msa.diversity for msa in msa.msa_collection])

# split data into training and testing set (set random_state for reproducibility)
div_train, div_test, t_train, t_test = train_test_split(
    div, t, test_size=0.3, random_state=10)


#########################
# fit linear regression
#########################
reg = LinearRegression().fit(div_train, t_train)
omega = reg.coef_
intercept = reg.intercept_

t = np.arange(100)
t_est_train = []
for i in range(len(div_train)):
    t_est_train.append(intercept + np.matmul(div_train[i], omega))
    print(f"t = {t_train[i]};  t_est = {t_est_train[-1]}")
t_est_test = []
for i in range(len(div_test)):
    t_est_test.append(intercept + np.matmul(div_test[i], omega))
    print(f"t = {t_test[i]};  t_est = {t_est_test[-1]}")

fig1 = plt.figure()
plt.ylabel('t estimate')
plt.xlabel('t')
plt.title("OLS")
plt.scatter(t_train, t_est_train, color='tab:blue', label="training set")
plt.scatter(t_test, t_est_test, color='tab:green', label='testing set')
plt.plot(t, t, color="black")
plt.legend()
fig1.show()



