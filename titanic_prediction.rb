require 'pycall/import'
include PyCall::Import
pyimport "pandas", as: :pd

# kaggleのtitanicのデータで機械学習
# https://www.kaggle.com/c/titanic

# データの読み込み（トレーニングデータとテストデータにすでに分かれていることに注目）
df_train = pd.read_csv.('./data/train.csv')
df_test = pd.read_csv.('./data/test.csv')

# SexId を追加
sex_dict = PyCall::Dict.new({'male': 1, 'female': 0})
df_train['SexId'] = df_train['Sex'].map.(sex_dict)
df_test['SexId'] = df_test['Sex'].map.(sex_dict)

# FamilySize = SibSp + Parch
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

# Ageの欠損値保管
df_train['AgeNull'] = df_train['Age'].isnull.()
age_median = df_train['Age'].median.()
df_train['Age'].fillna.(age_median, inplace: true)
df_test['Age'].fillna.(age_median, inplace: true)

inputs = ['FamilySize', 'SexId', 'Age']

X_train = df_train[inputs].values.astype.('float32')
X_test = df_test[inputs].values.astype.('float32')

y_train = df_train['Survived'].values

# ランダムフォレスト
pyfrom 'sklearn.ensemble', import: 'RandomForestClassifier'
model = RandomForestClassifier.(random_state: 42)

# 学習
model.fit.(X_train, y_train)

# 予測
score = model.score.(X_train, y_train)
puts "RandomForestClassifier pred: #{score}"

# 正規化したデータをCSVで残しておく
# df_train.to_csv.('data/train_formatted.csv')
# df_test.to_csv.('data/test_formatted.csv')

# シリアライズ(保存)
# pyfrom 'sklearn.externals', import: 'joblib'
# joblib.dump.(model, 'model/rf.pkl')
