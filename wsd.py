import time

import tensorflow as tf
from model import Model


def installation_test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(str(sess.run(hello)))
    import platform
    print("INFO: Python version: ", platform.sys.version)
    print("INFO: Tensorflow version: ", str(tf.VERSION))
    print("INFO: GPU found: ", tf.test.gpu_device_name())


def main(_):

    installation_test()

    model = Model()

    start = time.time()

    model.build_graph_and_train()

    print("Trained and evaluated in: ", time.time() - start)


if __name__ == "__main__":
    tf.app.run()
