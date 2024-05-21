import pandas as pd
import numpy as np

train = pd.read_csv("mercari-price-suggestion-challenge/train.tsv", delimiter='\t', low_memory=True)
test = pd.read_csv("mercari-price-suggestion-challenge/test.tsv", delimiter='\t', low_memory=True)

#trainデータ
train.name = train.name.astype("category")
train.category_name = train.category_name.astype("category")
train.brand_name = train.brand_name.astype("category")
train.item_description = train.item_description.astype("category")

#testデータ
test.name = test.name.astype("category")
test.category_name = test.category_name.astype("category")
test.brand_name = test.brand_name.astype("category")
test.item_description = test.item_description.astype("category")

train_test_combine = pd.concat([train.drop(["price"],axis=1), test], axis=0) #axis=0で行、axis=1で列に結合　デフォルトはaxis=0


train_test_combine.name = train_test_combine.name.astype("category")
train_test_combine.category_name = train_test_combine.category_name.astype("category")
train_test_combine.brand_name = train_test_combine.brand_name.astype("category")
train_test_combine.item_description = train_test_combine.item_description.astype("category")

train_test_combine.train_id = train_test_combine.train_id.fillna(pd.Series(train_test_combine.index))
train_test_combine.test_id = train_test_combine.test_id.fillna(pd.Series(train_test_combine.index))

train_test_combine.train_id = train_test_combine.train_id.astype(np.int64)
train_test_combine.test_id = train_test_combine.test_id.astype(np.int64)

train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
train_test_combine.item_description = train_test_combine.item_description.cat.codes

print('前処理完了"')

df_train = train_test_combine.iloc[:train.shape[0],:]
df_test = train_test_combine.iloc[train.shape[0]:,:]

#df_trainでtest_idを削除
df_train = df_train.drop(["test_id"], axis=1)
#df_testでtrain_idを削除
df_test = df_test.drop(["train_id"], axis=1)

df_test = df_test[["test_id"] + [col for col in df_test.columns if col != "test_id"]]

df_train["price"] = train.price
print("学習開始")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_valid, y_train, y_valid = train_test_split(df_train.drop(["price"], axis=1), df_train.price, test_size=0.2, random_state=42)

clf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
