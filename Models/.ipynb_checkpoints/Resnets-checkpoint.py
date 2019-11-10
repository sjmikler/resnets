import tensorflow as tf


def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, padding='same', kernel_regularizer=_regularizer,
                                  kernel_initializer='he_normal')


def bn_relu(x):
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def shortcut(x, filters, stride, mode):
    if x.shape[-1] == filters:
        return x
    if mode == 'preactivated_projection':
        return regularized_padded_conv(filters, kernel_size=1, strides=stride)(bn_relu(x))
    if mode == 'B' or mode == 'projection':
        return regularized_padded_conv(filters, kernel_size=1, strides=stride)(x)
    if mode == 'A' or mode == 'padding':
        return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride>1 else x,
                      paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])


def original_simple_block(x, filters, stride=1):
    c1 = regularized_padded_conv(filters, kernel_size=3, strides=stride)(x)
    c2 = regularized_padded_conv(filters, kernel_size=3)(bn_relu(c1))
    c2 = tf.keras.layers.BatchNormalization()(c2)
    
    x = shortcut(x, filters, stride, mode=_shortcut_mode)
    return tf.keras.layers.ReLU()(tf.add(x, c2))
    
    
def preactivation_block(x, filters, stride=1):
    global _omit_first_activation
    if _omit_first_activation:
        _omit_first_activation = False
        flow = x
    else:
        flow = bn_relu(x)
        
    c1 = regularized_padded_conv(filters, kernel_size=3, strides=stride)(flow)
    c2 = regularized_padded_conv(filters, kernel_size=3)(bn_relu(c1))
    
    x = shortcut(x, filters, stride, mode=_shortcut_mode)
    return tf.add(x, c2)


def bootleneck_block(x, filters, stride=1):
    global _omit_first_activation
    if _omit_first_activation:
        _omit_first_activation = False
        flow = x
    else:
        flow = bn_relu(x)
        
    c1 = regularized_padded_conv(filters//4, kernel_size=1)(flow)
    c2 = regularized_padded_conv(filters//4, kernel_size=3, strides=stride)(bn_relu(c1))
    c3 = regularized_padded_conv(filters, kernel_size=1)(bn_relu(c2))
    
    x = shortcut(x, filters, stride, mode=_shortcut_mode)
    return tf.add(x, c3)


def group_of_blocks(x, block_type, num_blocks, filters, stride):
    x = block_type(x, filters, stride)
    for i in range(num_blocks-1):
        x = block_type(x, filters)
    return x


def Resnet(input_shape, n_classes, weight_decay=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
          shortcut_mode='B', block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1}):
    
    global _regularizer, _shortcut_mode, _omit_first_activation
    _regularizer = tf.keras.regularizers.l2(weight_decay)
    _shortcut_mode = shortcut_mode
    
    if block_type == 'preactivated':
        selected_block = preactivation_block
        _omit_first_activation = True
    elif block_type == 'bootleneck':
        selected_block = bootleneck_block
        _omit_first_activation = True
    elif block_type == 'original':
        selected_block = original_simple_block
    else:
        raise KeyError("Parameter block_type not recognized!")
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = regularized_padded_conv(**first_conv)(inputs)
    flow = bn_relu(flow)
    
    for group_size, feature, stride in zip(group_sizes, features, strides):
        flow = group_of_blocks(flow, block_type=selected_block, num_blocks=group_size, filters=feature, stride=stride)
    
    if block_type != 'original':
        flow = bn_relu(flow)
    
    flow = tf.keras.layers.GlobalAveragePooling2D()(flow)
    outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def cifar_resnet20(shortcut_mode='B', block_type='preactivated'):
    return Resnet(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(3, 3, 3), features=(16, 32, 64),
                 strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_mode=shortcut_mode, block_type=block_type)


def cifar_resnet32(shortcut_mode='B', block_type='preactivated'):
    return Resnet(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(5, 5, 5), features=(16, 32, 64),
                 strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_mode=shortcut_mode, block_type=block_type)


def cifar_resnet44(shortcut_mode='B', block_type='preactivated'):
    return Resnet(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(7, 7, 7), features=(16, 32, 64),
                 strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_mode=shortcut_mode, block_type=block_type)


def cifar_resnet56(shortcut_mode='B', block_type='preactivated'):
    return Resnet(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(9, 9, 9), features=(16, 32, 64),
                 strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_mode=shortcut_mode, block_type=block_type)


def cifar_resnet110(shortcut_mode='B', block_type='preactivated'):
    return Resnet(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(18, 18, 18), features=(16, 32, 64),
                 strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_mode=shortcut_mode, block_type=block_type)


def cifar_resnet164(shortcut_mode='preactivated_projection', block_type='bootleneck'):
    return Resnet(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(18, 18, 18), features=(64, 128, 256),
                 strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_mode=shortcut_mode, block_type=block_type)


def cifar_resnet1001(shortcut_mode='preactivated_projection', block_type='bootleneck'):
    return Resnet(input_shape=(32, 32, 3), n_classes=10, weight_decay=1e-4, group_sizes=(111, 111, 111), features=(64, 128, 256),
                 strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_mode=shortcut_mode, block_type=block_type)