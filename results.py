from tensorflow.keras.models import load_model
from utils import plot_hist, predict, get_image_data
import pandas as pd
import numpy as np


# recognized emotions
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# fetching separated test data
with open('test.npy', 'rb') as f:
    x_test = np.load(f)
    y_test = np.load(f)

# read history od training
df = pd.read_csv('history.csv')
plot_hist(df)

# read model
model = load_model('my_model3.h5')

print("Generate predictions for x_test samples")
predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)


for i in range(len(x_test)):
    print(label_map[predictions[i].tolist().index(max(predictions[i]))])

predict(predictions, label_map, x_test)