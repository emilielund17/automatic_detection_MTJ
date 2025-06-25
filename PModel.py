import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# def load_data(image_path, label_path):
#     X = np.load(image_path)
#     y = np.load(label_path)
    
#     height = 15
#     width = 15
#     height_vec = np.zeros(y.shape[0])
#     width_vec = np.zeros(y.shape[0])
#     height_vec[:] = height
#     width_vec[:] = width
    
#     y = np.column_stack((y, height_vec, width_vec))
#     y -= [width // 2, height // 2, 0, 0]
#     y /= [X.shape[2], X.shape[1], X.shape[2], X.shape[1]]

#     return X, y

def load_data(image_path, label_path):
    X = np.load(image_path)
    Y = np.load(label_path)

    Y = Y.astype(np.float32)
    img_height, img_width = X.shape[1], X.shape[2]
    default_width, default_height = 50, 40  # Fixed size bounding box assumption

    # Create width and height vectors
    width_vec = np.full(Y.shape[0], default_width)
    height_vec = np.full(Y.shape[0], default_height)
    
    # Stack to form [x, y, w, h]
    Y = np.column_stack((Y, width_vec, height_vec))

    # Convert to center-based coordinates
    Y[:, 0] -= default_width // 2  # Adjust x
    Y[:, 1] -= default_height // 2  # Adjust y

    # Normalize bounding boxes
    Y[:, 0] /= img_width   # Normalize x (center)
    Y[:, 1] /= img_height  # Normalize y (center)
    Y[:, 2] /= img_width   # Normalize width
    Y[:, 3] /= img_height  # Normalize height

    return X, Y

def denormalize_bbox(bbox, img_width, img_height):
    """Convert normalized bbox back to image coordinates."""
    x, y, w, h = bbox
    x = int(x * img_width)   
    y = int(y * img_height)
    w = int(w * img_width)  
    h = int(h * img_height)  
    return x, y, w, h
    

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding="same", kernel_regularizer=l2(0.001))(x)
        shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(filters, kernel_size, strides=strides, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters, kernel_size, strides=(1, 1), padding="same", kernel_regularizer=l2(0.001))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation("relu")(x) 

    return x

def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2)  # Output layer for bounding box (x, y, width, height)
    ])
    return model

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (10, 10), padding="same", activation="relu", kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (10, 10), padding="same", activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # x = residual_block(x, filters=16)
    # x = residual_block(x, filters=16)
    # x = residual_block(x, filters=32)
    # x = residual_block(x, filters=32)
    # x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(2, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model

if __name__ == '__main__':
    path = ['all_data.npy', 'all_ground_truth.npy']
    images, labels = load_data(path[0], path[1])

    # Get the height and width of the images
    img_height, img_width = images.shape[1], images.shape[2]
    print(f"Image height: {img_height}, Image width: {img_width}")

    input_shape = (img_height, img_width, 1)
    model = build_model(input_shape=input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    
    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    checkpoint_callback = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_mae',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(monitor='val_mae', patience=7, restore_best_weights=True)
    
    history = model.fit(
        X_train, Y_train,
        epochs=15,
        batch_size=16,
        validation_data=(X_val, Y_val),
        callbacks=[checkpoint_callback, early_stopping],
        verbose=1
    )
    
    # # Plot the training and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()  

    # Load the trained model
    model.load_weights('best_model_bbox.h5')
    
    # Evaluate on the validation set
    eval_loss, eval_mae = model.evaluate(X_val, Y_val, verbose=1)

    print(f"Validation Loss: {eval_loss:.4f}, Validation MAE: {eval_mae:.4f}")

        # Save validation results to a .txt file
    with open("model_metrics.txt", "a") as f:
        f.write(f"\n--- TEST ---\n")
        f.write(f"Validation Loss: {eval_loss:.4f}, Validation MAE: {eval_mae:.4f}\n")
        f# Print five random predictions with ground truth
        f.write("\nRandom Sample Predictions vs Ground Truth:\n")
        for _ in range(5):
            idx = np.random.randint(0, len(images))
            sample_gt = labels[idx]
            sample_pred = model.predict(images[idx][np.newaxis, ...])[0]

            # Denormalize bounding boxes
            gt_x, gt_y, gt_w, gt_h = denormalize_bbox(sample_gt, img_width, img_height)
            pr_x, pr_y, pr_w, pr_h = denormalize_bbox(sample_pred, img_width, img_height)

            f.write(f"\nSample {idx}:\n")
            f.write(f"  Ground Truth: x={gt_x}, y={gt_y}, w={gt_w}, h={gt_h}\n")
            f.write(f"  Prediction  : x={pr_x}, y={pr_y}, w={pr_w}, h={pr_h}\n")

        f.write("-" * 40 + "\n")

    # Select a random sample for visualization
    idx = np.random.randint(0, len(images))
    sample_image = images[idx]
    ground_truth = labels[idx]
    
    # Model prediction
    predicted_bbox = model.predict(sample_image[np.newaxis, ...])[0] 
    print("Predicted Bounding Box (normalized):", predicted_bbox)

    # Denormalize bounding boxes
    gt_x, gt_y, gt_w, gt_h = denormalize_bbox(ground_truth, img_width, img_height)
    pr_x, pr_y, pr_w, pr_h = denormalize_bbox(predicted_bbox, img_width, img_height)
    print(f"Ground truth bbox: x={gt_x}, y={gt_y}, w={gt_w}, h={gt_h}")
    print(f"Predicted bbox (denormalized): x={pr_x}, y={pr_y}, w={pr_w}, h={pr_h}")

    # Plot image with bounding boxes
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(sample_image, cmap='gray')

    # Ground truth bounding box (Green)
    rect_gt = plt.Rectangle((gt_x - gt_w//2, gt_y - gt_h//2), gt_w, gt_h, 
                        linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth')
    ax.add_patch(rect_gt)

    # Plot predicted bounding box if valid
    if pr_w > 0 and pr_h > 0:
        rect_pred = plt.Rectangle((pr_x - pr_w//2, pr_y - pr_h//2), pr_w, pr_h, 
                              linewidth=2, edgecolor='red', facecolor='none', label='Prediction')
        ax.add_patch(rect_pred)
    else:
        print("Warning: Predicted bounding box has invalid dimensions")

    ax.set_title("Bounding Box Prediction vs Ground Truth")
    ax.legend()
    plt.show()


    ################################ Patricks Code ################################
    # input_shape = (height, width, 1)
    # model = build_cnn(input_shape)
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # # Split data into training and validation sets
    # split_index = int(0.8 * len(images))
    # train_images, val_images = images[:split_index], images[split_index:]
    # train_labels, val_labels = labels[:split_index], labels[split_index:]

    # checkpoint_callback = ModelCheckpoint(
    #     filepath='best_model.h5',  # Path where the model will be saved
    #     monitor='val_loss',        # Metric to monitor
    #     save_best_only=True,       # Save only the best model
    #     mode='min',                # Mode to determine the best model (min for loss)
    #     verbose=1                  # Verbosity mode
    # )

    # # Train the model
    # model.fit(
    #     train_images, train_labels,
    #     epochs=10,
    #     batch_size=32,
    #     validation_data=(val_images, val_labels),
    #     callbacks=[checkpoint_callback]  # Add the callback here
    # )

    # # Plot the training and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(model.history['loss'], label='Training Loss')
    # plt.plot(model.history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # # plt.savefig('loss_plot.png')  # Save the plot as a PNG file
    # plt.show()