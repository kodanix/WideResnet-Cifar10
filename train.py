from argparse import ArgumentParser
import os
import tensorflow as tf
import numpy as np
from dataset import CIFAR10
from models import WideResidualNetwork
import util

def scheduler(epoch, lr):
    learningRateDecay = 0.2
    if epoch == 60:
        return float(lr * learningRateDecay) 
    elif epoch == 120:
        return float(lr * learningRateDecay) 
    elif epoch == 160:
        return float(lr * learningRateDecay) 
    else:
        return float(lr)

def main(params):
    ds = CIFAR10()
    x_train, y_train = ds.get_train()
    x_val, y_val = ds.get_val()
    print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))
    model_holder = WideResidualNetwork()
    model = model_holder.build_model(ds.get_input_shape(), ds.get_nb_classes())
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    model.compile(tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True, name='SGD'), loss_fn, metrics)
    m_path = os.path.join(params.save_dir, model_holder.get_name())
    util.mk_parent_dir(m_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + '_{epoch:03d}-{val_loss:.2f}.h5'),
                 tf.keras.callbacks.CSVLogger(os.path.join(params.save_dir, 'metrics.csv')),
                 tf.keras.callbacks.LearningRateScheduler(scheduler)]
    model.fit(x_train, y_train, epochs=params.epoch, validation_data=(x_val, y_val),
              batch_size=params.batch_size,
              callbacks=callbacks, verbose=0)


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--memory_limit", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default=os.path.join('saved_models'))
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        selected = gpus[FLAGS.gpu]
        tf.config.experimental.set_visible_devices(selected, 'GPU')
        tf.config.experimental.set_memory_growth(selected, True)
        tf.config.experimental.set_virtual_device_configuration(
            selected,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.memory_limit)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        l_gpu = logical_gpus[0]
        with tf.device(l_gpu.name):
            main(FLAGS)
    else:
        main(FLAGS)