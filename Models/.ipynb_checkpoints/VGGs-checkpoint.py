import tensorflow as tf


def regularized_padded_conv3(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, kernel_size=3, kernel_regularizer=_regularizer,
                                  padding='same', kernel_initializer='he_normal')


def bn_relu(x):
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def conv_block(x, num_blocks, filters, pool):
    x = regularized_padded_conv3(filters=filters)(x)
    for i in range(num_blocks-1):
        x = regularized_padded_conv3(filters=filters)(bn_relu(x))
    x = tf.keras.layers.MaxPool2D(pool)(bn_relu(x))
    return x


def VGG(input_shape, n_classes, weight_decay=1e-4, group_sizes=(1, 1, 2, 2, 2),
           features=(64, 128, 256, 512, 512), pools=(2, 2, 2, 2, 2)):
    global _regularizer
    _regularizer = tf.keras.regularizers.l2(weight_decay)
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    flow = inputs
    for group_size, feature, pool in zip(group_sizes, features, pools):
        flow = conv_block(flow, num_blocks=group_size, filters=feature, pool=pool)
    
    pooled = tf.keras.layers.GlobalAveragePooling2D()(flow)
    output = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def cifar_vgg11():
    return VGG(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(1, 1, 2, 2, 2), 
               features=(64, 128, 256, 512, 512), pools=(2, 2, 2, 2, 2))


def cifar_vgg16():
    return VGG(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(2, 2, 3, 3, 3), 
               features=(64, 128, 256, 512, 512), pools=(2, 2, 2, 2, 2))


def cifar_vgg19():
    return VGG(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(2, 2, 4, 4, 4), 
               features=(64, 128, 256, 512, 512), pools=(2, 2, 2, 2, 2))