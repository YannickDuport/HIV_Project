
import pandas as pd

# dataset
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



# load dataset
house_price = load_boston()
print(house_price)
df = pd.DataFrame(house_price.data, columns=house_price.feature_names)
df['PRICE'] = house_price.target
print(df)

# standardize and train/test split
print(house_price.data)
house_price.data = preprocessing.scale(house_price.data)
print(house_price.data)
X_train, X_test, y_train, y_test = train_test_split(
    house_price.data, house_price.target, test_size=0.3, random_state=10)