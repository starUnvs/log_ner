from glob import glob
import pandas as pd

sample_num = 20000
file_paths = glob('./*.csv')

trains = []
tests = []
for path in file_paths:
    df = pd.read_csv(path).drop_duplicates().sample(sample_num)
    train = df.sample(frac=0.8)
    test = pd.concat([train, df]).drop_duplicates(keep=False)

    trains.append(train)
    tests.append(test)

pd.concat(trains).to_csv('./train_full.csv')
pd.concat(tests).to_csv('./test_full.csv')

pass
