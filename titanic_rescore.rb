require 'pycall/import'
include PyCall::Import
pyfrom 'sklearn.externals', import: 'joblib'
pyimport "pandas", as: :pd

# データの読み込み
df_train = pd.read_csv.('./data/train_formatted.csv')
inputs = ['FamilySize', 'SexId', 'Age']
X_train = df_train[inputs].values.astype.('float32')
y_train = df_train['Survived'].values

# 予測モデルを復元
model = joblib.load.('model/rf.pkl')

# 予測
score = model.score.(X_train, y_train)
puts "RandomForestClassifier pred: #{score}"
