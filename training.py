from sklearn.linear_model import LogisticRegression
import numpy as np
    
thumb_up_data = np.loadtxt('up_data.csv', delimiter=',')
thumb_down_data = np.loadtxt('down_data.csv', delimiter=',')
thumb_right_data = np.loadtxt('right_data.csv', delimiter=',')
thumb_left_data = np.loadtxt('left_data.csv', delimiter=',')
thumb_open_data = np.loadtxt('open_data.csv', delimiter=',')
thumb_close_data = np.loadtxt('close_data.csv', delimiter=',')

X = np.concatenate([thumb_up_data, thumb_down_data, thumb_right_data, thumb_left_data, thumb_open_data, thumb_close_data])

y = np.concatenate([np.ones(len(thumb_up_data)), np.zeros(len(thumb_down_data)), np.ones(len(thumb_right_data)) * 2, np.ones(len(thumb_left_data)) * 3, np.ones(len(thumb_open_data)) * 4, np.ones(len(thumb_close_data)) * 5])

model = LogisticRegression()
model.fit(X, y)

import pickle
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)