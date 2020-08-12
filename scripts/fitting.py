from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import lasso_path, enet_path

from scripts.msa import MSA_collection
from scripts.helpers import DATA_PATH

data_train_path = DATA_PATH / "fasta" / "training"
data_test_path = DATA_PATH / "fasta" / "testing"

# create MSA
msa_train = MSA_collection(data_train_path)
msa_test = MSA_collection(data_test_path)

# extract time from filenames
t_train = [int(name.stem.split('_')[-1][1:]) for name in list(data_train_path.glob('*.fasta'))]
t_test = [int(name.stem.split('_')[-1][1:]) for name in list(data_test_path.glob('*.fasta'))]
t_train.sort()
t_test.sort()

# calculate diversities
div_train = np.array([msa.diversity for msa in msa_train.msa_collection])
div_test = np.array([msa.diversity for msa in msa_test.msa_collection])


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
plt.title("linear regression fit")
plt.scatter(t_train, t_est_train, color='tab:blue', label="training set")
plt.scatter(t_test, t_est_test, color='tab:green', label='testing set')
plt.plot(t, t, color="black")
plt.legend()
fig1.show()



