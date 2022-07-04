import numpy as np
import pandas as pd
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
np.set_printoptions(precision=6, suppress=True)
from utils import get_image_data, create_model

x_train, y_train, x_valid, y_valid = get_image_data()

# dimensions of images
img_width = 48
img_height = 48
img_depth = 1

input_shape = (48, 48, 1)

model = create_model(img_width, img_height, img_depth)

early_stopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.00005, patience=11, verbose=1,
                               restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=7, min_lr=1e-7, verbose=1)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
history = model.fit(
    np.array(x_train),
    np.array(y_train),
    batch_size=64,
    epochs=60,
    verbose=1,
    validation_data=(np.array(x_valid), np.array(y_valid)),
    callbacks=[lr_scheduler, early_stopping]
)

# save history to csv file
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# train model and save to file
score = model.evaluate(np.array(x_valid), np.array(y_valid))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('my_model3.h5')










