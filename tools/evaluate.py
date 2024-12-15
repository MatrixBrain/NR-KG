import os
import numpy as np
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def evaluate(ground_truth, out, res_path, i, args, name='test'):
    # Calculate the MSE, MAE, RMSE, and R2 of the data
    mse = mean_squared_error(ground_truth, out)
    mae = mean_absolute_error(ground_truth, out)
    r2 = r2_score(ground_truth, out)

    # Print the results
    print('Evaluation results - fold: {:02d}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}'.format(i, name + '_mse', mse, name + '_mae', mae, name + '_r2', r2))

    # Save the results to root_path
    if not os.path.exists(res_path):
        with open(res_path, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['fold', name + '_mse', name + '_mae', name + '_r2'])
    with open(res_path, 'a+', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([i, mse, mae, r2])

    if i >= args.n_fold - 1:
        # Calculate the mean of the results
        with open(res_path, 'r', newline='') as f:
            reader_csv = csv.reader(f)
            data = list(reader_csv)
            data = np.array(data[1:], dtype=np.float32)
            data = data.mean(axis=0)
        with open(res_path, 'a+', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['mean', data[1], data[2], data[3]])
