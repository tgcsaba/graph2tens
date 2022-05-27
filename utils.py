from tensorflow.keras.callbacks import EarlyStopping

class EarlyStoppingAlwaysRestore(EarlyStopping):
    """
    A small upgrade on the standard EarlyStopping callback of keras. Fixes the issue that early stopping does not restore the best weights with the
    restore_best_weights option when the optimization ends by reaching the max number of epochs (rather than early stopping triggering the end of training).
    See:
        https://github.com/keras-team/keras/issues/12511
        https://github.com/tensorflow/tensorflow/issues/35634
    """  
    def on_train_end(self, logs=None):
        EarlyStopping.on_train_end(self, logs=logs)
        if self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)