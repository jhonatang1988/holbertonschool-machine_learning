#!/usr/bin/env python3
"""
performs forward propagation over a pooling layer of a neural network:
"""
import numpy as np

"""
A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
m is the number of examples
h_prev is the height of the previous layer
w_prev is the width of the previous layer
c_prev is the number of channels in the previous layer
kernel_shape is a tuple of (kh, kw) containing the size of the kernel for the pooling
kh is the kernel height
kw is the kernel width
stride is a tuple of (sh, sw) containing the strides for the pooling
sh is the stride for the height
sw is the stride for the width
mode is a string containing either max or avg, indicating whether to perform maximum or average pooling, respectively
you may import numpy as np
Returns: the output of the pooling layer
"""


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network
    """
    # tres paquetes para cada sample
    imagenes = A_prev
    num_imagenes, altura_img, largo_img, num_canales_img = imagenes.shape
    # W se refiere al filtro
    altura_filtro, largo_filtro = kernel_shape
    # stride se refiere a pasos
    altura_stride, largo_stride = stride
    
    # se calcula el tamano del output y se inicializa
    altura_output = int(
        ((altura_img - altura_filtro) / altura_stride) + 1)
    largo_output = int(
        ((largo_img - largo_filtro) / largo_stride) + 1)
    output = np.zeros(
        (num_imagenes, altura_output, largo_output, num_canales_img))
    
    # para cada stride moviendose verticalmente
    for vertical_stride in range(altura_output):
        # para cada stride moviendose horizontalmente
        for horizontal_stride in range(largo_output):
            # para cada filtro del output
            va = vertical_stride * altura_stride
            vaa = vertical_stride * altura_stride + altura_filtro
            hl = horizontal_stride * largo_stride
            hll = horizontal_stride * largo_stride + largo_filtro
            pedacito_imagen = imagenes[:, va:vaa, hl:hll]
            # el pooling como tal - numero_maximo_en_el_pedacito_de_imagen
            nmeepdi = np.max(pedacito_imagen, axis=(1, 2))
            output[:, vertical_stride, horizontal_stride] = \
                nmeepdi
    
    return output
