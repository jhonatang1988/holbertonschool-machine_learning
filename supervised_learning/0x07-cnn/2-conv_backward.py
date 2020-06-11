#!/usr/bin/env python3
"""
performs back propagation over a convolutional layer of a neural network
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the unactivated output of the
    convolutional layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
    kh is the filter height
    kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    padding is a string that is either same or valid, indicating the type of
    padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    you may import numpy as np
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    # toda esta mierda para que no pep no haga complaint por no usar la
    # variable
    biases = b
    if biases == 'hola':
        pass
    # la parte de las derivadas, sin aplicar la activacion porque tambien
    # toca derivar por aparte la puta funcion de activacion
    derivadas_parciales_layer_sin_activacion = dZ
    num_derivadas, altura_layer, largo_layer, num_canales_layer = \
        derivadas_parciales_layer_sin_activacion.shape
    
    # el layer previo OJO es el siguiente, pero vamos hacia el pasado.
    layer_previa = A_prev
    
    num_imagenes_layer_previa, altura_layer_previa, _, _ = layer_previa.shape
    _, _, largo_layer_previa, num_canales_layer_previa = layer_previa.shape
    
    # los filtros tambien se derivan pero es mas como una convolution en
    # reversa segun los expertos
    filtros = W
    altura_filtro, largo_filtro, _, _ = filtros.shape
    _, _, num_canales_filtro_previa, num_canales_filtro_nuevo = filtros.shape
    
    # el stride de toda la vida
    altura_stride, largo_stride = stride
    
    # para aplicar el padding correct
    if type(padding) is tuple:
        altura_padding, largo_padding = padding
    elif padding == 'same':
        altura_padding = int(
            np.ceil(
                ((altura_layer_previa - 1) * altura_stride + altura_filtro -
                 altura_layer_previa) / 2))
        largo_padding = int(
            np.ceil(((largo_layer_previa - 1) * largo_stride + largo_filtro -
                     largo_layer_previa) / 2))
    else:
        altura_padding, largo_padding = (0, 0)
    
    activaciones_con_padding = np.pad(layer_previa, ((0, 0), (altura_padding,
                                                              altura_padding),
                                                     (largo_padding,
                                                      largo_padding), (0, 0)),
                                      'constant')
    
    # sacamos la matrix de las derivadas de las activacions ie outputs.
    derivadas_de_activaciones = np.zeros(layer_previa.shape)
    
    # la puta derivada tambien lleva padding
    derivadas_de_activaciones_con_padding = np.pad(
        derivadas_de_activaciones, ((0, 0), (altura_padding, altura_padding),
                                    (largo_padding, largo_padding),
                                    (0, 0)))
    
    # toca una matrix para meter las derivadas de los filtros. El tamano es
    # la misma mierda que los filtros, obveoo bobis
    derivadas_de_filtros = np.zeros(filtros.shape)
    
    # por ultimo la derivada de los biases
    derivadas_de_biases = np.sum(derivadas_parciales_layer_sin_activacion,
                                 axis=(0, 1, 2),
                                 keepdims=True)
    
    # para cada imagen
    for index_imagen in range(num_imagenes_layer_previa):
        imagen = activaciones_con_padding[index_imagen]
        derivada_de_la_imagen = derivadas_de_activaciones_con_padding[
            index_imagen]
        # moviendonos verticalmente
        for vertical_stride in range(altura_layer):
            # moviendonos horizontalmente
            for horizontal_stride in range(largo_layer):
                # y en cada canal del layer
                for index_canal in range(num_canales_layer):
                    va = vertical_stride * altura_stride
                    vaa = vertical_stride * altura_stride + altura_filtro
                    hl = horizontal_stride * largo_stride
                    hll = horizontal_stride * largo_stride + largo_filtro
                    
                    pedacito_imagen = imagen[va:vaa, hl:hll]
                    
                    el_filtro = filtros[:, :, :, index_canal]
                    pedacito_derivadas = \
                        derivadas_parciales_layer_sin_activacion[
                            index_imagen,
                            vertical_stride,
                            horizontal_stride,
                            index_canal]
                    derivadas_por_filtros = el_filtro * pedacito_derivadas
                    
                    # guardamos en la matrix de derivadas
                    derivada_de_la_imagen[va:vaa, hl:hll] += \
                        derivadas_por_filtros
                    
                    # guardamos en la matrix de derivadas de filtros,
                    # pues las putas derivadas de filtros
                    derivadas_de_filtros[:, :, :, index_canal] += \
                        pedacito_imagen * \
                        derivadas_parciales_layer_sin_activacion[
                            index_imagen, vertical_stride,
                            horizontal_stride, index_canal]
        
        if padding == 'same':
            derivadas_de_activaciones[index_imagen] += \
                derivada_de_la_imagen[
                altura_padding: -altura_padding,
                largo_padding:-largo_padding]
        else:
            derivadas_de_activaciones[index_imagen] += derivada_de_la_imagen
    
    return derivadas_de_activaciones, derivadas_de_filtros, derivadas_de_biases
