import tensorflow as tf


def exp_decay(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_lrscheduler(model, epochs, train_data, val_data, scheduler):
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(
        train_data, validation_data=val_data, epochs=epochs, callbacks=[callback]
    )
    return history


def train(model, epochs, train_data, val_data):
    history = model.fit(train_data, validation_data=val_data, epochs=epochs)
    return history
