#!/usr/bin/env python3
"""
performs forward propagation over a convolutional layer of a neural network
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer of a neural network
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    :param W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution
    kh is the filter height
    kw is the filter width
    c_prev is the number of channels in the previous layer
    c_new is the number of channels in the output
    :param b: numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution
    :param activation: an activation function applied to the convolution
    :param padding: a string that is either same or valid, indicating the
    type of padding used
    :param stride: a tuple of (sh, sw) containing the strides for the
    convolution
    :return: the output of the convolutional layer
    """
    # tres paquetes para cada sample
    imagenes = A_prev
    num_imagenes, \
    altura_img, \
    largo_img, \
    num_canales_img = imagenes.shape
    
    # W se refiere al filtro
    filtros = W
    altura_filtro, \
    largo_filtro, \
    num_filtros, \
    num_filtros_nuevos = filtros.shape
    
    # stride se refiere a pasos
    altura_stride, largo_stride = stride
    
    # para aplicar el padding correct
    if type(padding) is tuple:
        altura_padding, largo_padding = padding
    elif padding == 'same':
        altura_padding = int(
            np.ceil(((altura_img - 1) * altura_stride + altura_filtro -
                     altura_img) / 2
                    )
        )
        largo_padding = int(
            np.ceil(((largo_img - 1) * largo_stride + largo_filtro -
                     largo_img) / 2
                    )
        )
    else:
        altura_padding, largo_padding = (0, 0)
    
    # se calcula el tamano del output y se inicializa
    altura_output = int(
        ((altura_img + (2 * altura_padding) - altura_filtro) /
         altura_stride) + 1
    )
    
    largo_output = int(
        ((largo_img + (2 * largo_padding) - largo_filtro) / largo_stride + 1
         )
    )
    
    output = np.zeros(
        (num_imagenes, altura_output, largo_output, num_filtros_nuevos))
    
    # ahora se aplica padding en los bordes, llenando de ceros la
    # informacion para que no se pierda la importancia de los bordes cuando
    # se hace convolution
    imagenes_con_padding = np.pad(imagenes, ((0, 0), (altura_padding,
                                                      altura_padding),
                                             (largo_padding, largo_padding),
                                             (0, 0)),
                                  'constant')
    
    # para cada stride moviendose verticalmente
    for vertical_stride in range(altura_output):
        # para cada stride moviendose horizontalmente
        for horizontal_stride in range(largo_output):
            # para cada filtro del output
            for filtro in range(num_filtros_nuevos):
                va = vertical_stride * altura_stride
                vaa = vertical_stride * altura_stride + altura_filtro
                hl = horizontal_stride * largo_stride
                hll = horizontal_stride * largo_stride + largo_filtro
                pedacito_imagen = imagenes_con_padding[:, va:vaa, hl:hll]
                # la convolution como tal
                pedacito_imagen_por_filtro = \
                    filtros[:, :, :, filtro] * pedacito_imagen
                
                # esa misma mierda se suma en todas las inner dimensions de
                # un 4d, es decir, suma en 1, 2, y 3, pero no suma en la 0,
                # que seria la mas outer, que corresponde al numero de
                # samples. OJO  la dimension mas inner es la mas ejemplo
                # aqui es la 3. putamente contra intruitivo
                # pipfsetld =
                # pedacito_imagen_por_filtro_sumados_en_todas_las_dimensiones
                pipfsetld = pedacito_imagen_por_filtro.sum(axis=(1, 2, 3))
                
                output[:, vertical_stride, horizontal_stride, filtro] = \
                    pipfsetld
    
    funcion_activacion = activation
    
    return funcion_activacion(output + b)
