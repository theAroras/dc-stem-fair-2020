import csv, string, re
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

X_dict, Y_dict = {}, {}
X_min, X_max = [float('inf')] * 200, [float('-inf')] * 200
heading = []

print("Reading data...")
with open('Well Being Index.csv') as fin:
    csv = csv.reader(fin, delimiter=',')
    for i, row in enumerate(csv):
        if i == 0:
            heading = row[31:]
            continue

        city_out = row[0]
        city_in = row[13]

        if city_in:
            city = re.sub(r'[^\w]', '', city_in)

            X_dict[city] = []
            for i in row[31:]:
                if i != '':
                    X_dict[city].append(float(i))
                else:
                    X_dict[city].append(None)

            for i, num in enumerate(X_dict[city]):
                if num:
                    X_min[i] = min(X_min[i], num)
                    X_max[i] = max(X_max[i], num)
            
        if city_out:
            city = re.sub(r'[^\w]', '', city_out)
            Y_dict[city] = float(row[2])
print("Successfully read all data.")

print("Parsing training data...")
X, Y = [], []
for city in X_dict:
    for i, num in enumerate(X_dict[city]):
        if num:
            try:
                X_dict[city][i] = ((X_dict[city][i] - X_min[i]) / (X_max[i] - X_min[i]))
            except:
                X_dict[city][i] = 0.0
        else:
            X_dict[city][i] = 0.5
    X.append(X_dict[city])
    Y.append(Y_dict[city])
print("Successfully parsed all training data.")

weights = [0.0] * 200

print("Running 100 trials...")
for i in range(100):
    model = MLPRegressor(max_iter=100000, hidden_layer_sizes=(), tol=1e-6)
    model.fit(X, Y)
    print("Trial #", i + 1)

    for i, j in enumerate(model.coefs_[0]):
        weights[i] += j[0] / 100

print("All trials done.")
print(weights)


