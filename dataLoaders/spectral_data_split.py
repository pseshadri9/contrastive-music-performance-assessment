# Generate train_test split for spectral dataset
import numpy as np
import dill

np.random.seed(1)

data_path = './dat/middle_2_data_3.dill'

perf_data = dill.load(open(data_path, 'rb'))
perf_data = np.array(perf_data)
ind = np.arange(1410)
np.random.shuffle(ind)

total = len(ind)
num_valid = int(total*0.1)
num_train = int(0.8*total)

for i in range(num_train):
	print(perf_data[ind[i]]['student_id'])
	

train_data = perf_data[ind[0:num_train]]
valid_data = perf_data[ind[num_train:num_train+num_valid]]
test_data = perf_data[ind[num_train+num_valid:num_train+2*num_valid]]

with open('./dat/' + 'middle_2_data_3_train.dill', 'wb') as f:
    dill.dump(train_data, f)
with open('./dat/' + 'middle_2_data_3_test.dill', 'wb') as f:
    dill.dump(test_data, f)
with open('./dat/' + 'middle_2_data_3_valid.dill', 'wb') as f:
    dill.dump(valid_data, f)

np.save('train', train_data)
np.save('test', test_data)
np.save('valid', valid_data)