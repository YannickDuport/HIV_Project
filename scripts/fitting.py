import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from scripts.msa import MSA_collection
from scripts.helpers import DATA_PATH

data_train_path = DATA_PATH / "fasta" / "training"
data_test_path = DATA_PATH / "fasta" / "testing"

# create MSA
msa_train = MSA_collection(data_train_path)
msa_test = MSA_collection(data_test_path)


# extract time from filenames
time_train = [int(name.stem.split('_')[-1][1:]) for name in list(data_train_path.glob('*.fasta'))]
time_test = [int(name.stem.split('_')[-1][1:]) for name in list(data_test_path.glob('*.fasta'))]
time_train.sort()
time_test.sort()

# plot diversity over time of first site
div_train = np.array([msa.diversity for msa in msa_train.msa_collection])
div_test= np.array([msa.diversity for msa in msa_test.msa_collection])
fig = plt.figure()
plt.plot(time_train, div_train.transpose()[0])

# fit linear regression
reg = LinearRegression().fit(div_train, time_train)
omega = reg.coef_
intercept = reg.intercept_
print(f"omega = {omega}")
print(f"intercept = {intercept}")

print('-'*80)
print("training data")
print('-'*80)
for i in range(len(div_train)):
    t_est = intercept + np.matmul(div_train[i], omega)
    print(f"t = {time_train[i]};  t_est = {t_est}")

print('\n')
print('-'*80)
print("testing data")
print('-'*80)
for i in range(len(div_test)):
    t_est = intercept + np.matmul(div_test[i], omega)
    print(f"t = {time_test[i]};  t_est = {t_est}")