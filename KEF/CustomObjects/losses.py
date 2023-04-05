import numpy
from sklearn import metrics
from keras import backend as K
import keras
import tensorflow as tf
from KEF.CustomObjects.vgg_face import face_embedding
from scipy.io import loadmat, savemat

flag = 0
def load_Vgg():
    # Please download vgg-face.mat from https://www.robots.ox.ac.uk/~vgg/software/vgg_face/ into the path below.
    vgg_weights = loadmat('KEF/Models/VGG/vgg-face.mat')
    return vgg_weights

def hinge_onehot(y_true, y_pred):
    y_true = y_true * 2 - 1
    y_pred = y_pred * 2 - 1

    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
def total_variation(y_true, y_pred):
    return tf.reduce_mean(tf.image.total_variation(y_pred))


def ID_Loss(y_true, y_pred):

    if flag == 0:
        vgg_weights = load_Vgg()

    real_conv1_2, real_conv2_2, real_conv3_2, real_conv4_2, real_conv5_2 = face_embedding(vgg_weights,
                                                                                          y_true)
    fake_conv1_2, fake_conv2_2, fake_conv3_2, fake_conv4_2, fake_conv5_2 = face_embedding(vgg_weights, y_pred)

    conv1_2_loss = tf.reduce_mean(tf.abs(real_conv1_2 - fake_conv1_2)) / 224. / 224. # dividing factor is VGG image size
    conv2_2_loss = tf.reduce_mean(tf.abs(real_conv2_2 - fake_conv2_2)) / 112. / 112.
    conv3_2_loss = tf.reduce_mean(tf.abs(real_conv3_2 - fake_conv3_2)) / 56. / 56.
    conv4_2_loss = tf.reduce_mean(tf.abs(real_conv4_2 - fake_conv4_2)) / 28. / 28.
    conv5_2_loss = tf.reduce_mean(tf.abs(real_conv5_2 - fake_conv5_2)) / 14. / 14.
    vgg_loss = conv1_2_loss + conv2_2_loss + conv3_2_loss + conv4_2_loss + conv5_2_loss

    return vgg_loss
    # -----------------------------------------------------------------------

def cyclic_loss_component(y_true, y_pred):

    return keras.losses.mean_absolute_error(y_true, y_pred)


def EG_loss(y_true, y_pred):
    loss =  0.5 * ID_Loss(y_true, y_pred) + cyclic_loss_component(y_true, y_pred) \
            + 0.000085 * total_variation(y_true, y_pred)
    return loss


def tanh_crossentropy(y_true, y_pred):
    y_true = 0.5 * (y_true + 1)
    y_pred = 0.5 * (y_pred + 1)
    return keras.losses.binary_crossentropy(y_true,y_pred)

def binary_crossentropy_with_logits(y_true, y_pred):

    return tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
def logistic_loss(y_true, y_pred):

    return K.mean(K.log(1+ K.exp(-1 * y_true * y_pred)) / numpy.log(2))

def hinge_onehot_squared(y_true, y_pred):
    y_true = y_true * 2 - 1
    y_pred = y_pred * 2 - 1

    return K.square(K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1))

def clipped_loss(y_true, y_pred):
    return K.mean(huber_loss(y_true, y_pred, clip_value=numpy.inf), axis=-1)

def huber_loss(y_true, y_pred, clip_value=0.5):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.

    assert clip_value > 0.

    x = y_true - y_pred
    if numpy.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

def variational_autoencoder_loss(y_true, y_pred, batchSize):
    generated_flat = K.reshape(y_pred, [batchSize, 28 * 28])

    generation_loss = - K.sum(y_true * K.log(1e-8 + generated_flat) + (1 - y_true) * K.log(1e-8 + 1 - y_pred), 1)
    return generation_loss


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)



