import tensorflow as tf
from model_tf import Model
from data_preprocessing import get_params_dict

def installation_test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(str(sess.run(hello)))
    print("INFO: Installed version: " + str(tf.VERSION))
    print("INFO: GPU found: ", tf.test.gpu_device_name())


def main(_):

    model = Model(get_params_dict())

    model.build_graph_and_train()


if __name__ == "__main__":
    tf.app.run()
