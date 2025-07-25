# 14-MLOps - MLFLOW

<img width="1105" height="999" alt="image" src="https://github.com/user-attachments/assets/ef638ebc-e337-4c4a-ab90-26cae868198b" />

<img width="539" height="284" alt="image" src="https://github.com/user-attachments/assets/067a33a0-b904-4440-8135-a410121224b6" />




#### mlflow components:

4 components:
1. tracking
2. projects
3. models
4. registry

   <img width="291" height="250" alt="image" src="https://github.com/user-attachments/assets/0f30fc07-e899-4fda-8e09-cfa95f205aa4" />

<img width="410" height="196" alt="image" src="https://github.com/user-attachments/assets/f94a4e72-ec6b-4fe4-8d02-18354ad40a5a" />

<img width="510" height="251" alt="image" src="https://github.com/user-attachments/assets/1239a14a-b012-4858-a6a3-8c93839b6c06" />

<img width="485" height="221" alt="image" src="https://github.com/user-attachments/assets/b2ab5e64-8ea3-4a9e-8850-06e77af0173c" />

<img width="486" height="198" alt="image" src="https://github.com/user-attachments/assets/68c5420e-b8cf-4987-8b03-064bdc231aa1" />

<img width="494" height="202" alt="image" src="https://github.com/user-attachments/assets/4b454e00-6d5a-42c4-ad16-f13a61820319" />

Install anaconda and pycharm in laptop


Pandas - to handle data frames

numpy - for mathematical computations.

```
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()

#evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("red-wine-quality.csv")
    data.to_csv("data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

```

