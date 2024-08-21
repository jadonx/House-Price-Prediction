# House-Price-Prediction
House Price Prediction Model with Machine Learning w/ Yanis

Source: https://www.youtube.com/watch?v=Wqmtf9SA_kk&t=1081s
Data: https://www.kaggle.com/datasets/camnugent/california-housing-prices

# Notes

### Reading Data
```Python
data = pd.read_csv("data.csv")
```

### Cleaning Data
```Python
# Dropping Null Values
data.dropna(inplace=True)
```

### Splitting Data (80% Train 20% Test)
```Python
from sklearn.model_selection import train_test_split

# Splitting x and y
x = data.drop(['median_house_value'], axis=1) # Input
y = data['median_house_value'] # Output

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# Recombine
train_data = x_train.join(y_train)
```