# House-Price-Prediction
House Price Prediction Model with Machine Learning w/ Yanis

Source: https://www.youtube.com/watch?v=Wqmtf9SA_kk&t=1081s
Data: https://www.kaggle.com/datasets/camnugent/california-housing-prices

# Notes
```Python
# Reading data
data = pd.read_csv("data.csv")

# Cleaning Data
# Dropping Null Values
data.dropna(inplace=True)

# Splitting Data (80% Train 20% Test)
from sklearn.model_selection import train_test_split

# Splitting x and y**
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# Recombine training data
train_data = x_train.join(y_train)
```