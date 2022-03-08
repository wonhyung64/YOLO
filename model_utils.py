#%%
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Add, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
#%%
def yolo_v3(input_shape, hyper_params, fine_tunning=True):
    base_model = DarkNet53(include_top=False, input_shape=input_shape)
    base_model.load_weights(os.getcwd() + '/darknet_weights/weights')#
    if fine_tunning == False: base_model.trainable=False

    total_labels = hyper_params["total_labels"]

    inputs = base_model.input
    c3, c2, c1 = base_model.output

    x = conv_block(c1, [{"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 75},
                        {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 76},
                        {"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 77},
                        {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 78},
                        {"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 79}
                        ], skip=False)

    head1 = conv_block(x, [{"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 80},
                            {"filter": 3 * (5 + total_labels), "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 81}
                            ], skip=False)

    x = conv_block(x, [{"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 84}], skip=False)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, c2])

    x = conv_block(x, [{"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 87},
                        {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 88},
                        {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 89},
                        {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 90},
                        {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 91}
                        ], skip=False)

    head2 = conv_block(x, [{"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 92},
                            {"filter": 3 * (5 + total_labels), "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 93}
                            ], skip=False)

    x = conv_block(x, [{"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 96}], skip=False)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, c3])

    x = conv_block(x, [{"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 99},
                        {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 100},
                        {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 101},
                        {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 102},
                        {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 103}
                        ], skip=False)

    head3 = conv_block(x, [{"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 104},
                            {"filter": 3 * (5 + total_labels), "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 105}
                            ], skip=False)
    
    return Model(inputs=inputs, outputs=[head1, head2, head3])

#%%
def DarkNet53(include_top=True, input_shape=(None, None, 3)):
    input_x = Input(shape=input_shape)

    x = conv_block(input_x, [{"filter": 32, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 0},
                                  {"filter": 64, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 1},
                                  {"filter": 32, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 2},
                                  {"filter": 64, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 3},
                                  ])

    x = conv_block(x, [{"filter": 128, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 5},
                        {"filter": 64, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 6},
                        {"filter": 128, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 7}
                        ])

    x = conv_block(x, [{"filter": 64, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 9},
                        {"filter": 128, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 10}
                        ])

    x = conv_block(x, [{"filter": 256, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 12},
                        {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 13},
                        {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 14}
                        ])

    for i in range(7):
        x = conv_block(x, [{"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 16+i*3},
                            {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 17+i*3}
                            ])
    c3 = x

    x = conv_block(x, [{"filter": 512, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 37},
                        {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 38},
                        {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 39}
                        ])

    for i in range(7):
        x = conv_block(x, [{"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 41+i*3},
                            {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 42+i*3}
                            ])
    c2 = x

    x = conv_block(x, [{"filter": 1024, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 62},
                        {"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 63},
                        {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 64}
                        ])

    for i in range(3):
        x = conv_block(x, [{"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 66+i*3},
                            {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 67+i*3}
                            ])
    c1 = x

    if include_top == False:
        return Model(inputs=input_x, outputs=[c3, c2, c1])
    x = GlobalAveragePooling2D(name="avgpooling")(x)
    output_x = Dense(1000, activation="softmax", name="fc")(x)

    return Model(inputs=input_x, outputs=output_x)

#%%
def conv_block(inp, convs, skip=True):
	x = inp
	count = 0
	for conv in convs:
		if count == (len(convs) - 2) and skip:
			skip_connection = x
		count += 1
		if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
		x = Conv2D(conv['filter'],
				   conv['kernel'],
				   strides=conv['stride'],
				   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
				   name='conv_' + str(conv['layer_idx']),
				   use_bias=False if conv['bnorm'] else True)(x)
		if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
		if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
	return Add()([skip_connection, x]) if skip else x

#%%
def yolo_head(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = tf.reshape(anchors, [1, 1, 1, num_anchors, 2])

    grid_shape = tf.shape(feats)[1:3]

    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])

    grid = tf.concat([grid_y, grid_x], axis=-1)
    grid = tf.cast(grid, tf.float32)

    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_yx = (tf.nn.sigmoid(feats[...,:2]) + grid) / tf.cast(grid_shape[...,-1], tf.float32)
    box_hw = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., -1], tf.float32)

    box_obj = tf.sigmoid(feats[..., 4:5])
    box_cls = tf.sigmoid(feats[..., 5:])

    return grid, feats, box_yx, box_hw, box_obj, box_cls
