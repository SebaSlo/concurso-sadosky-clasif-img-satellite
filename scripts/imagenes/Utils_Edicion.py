import cv2 as cv
import numpy as np

######################################################################################
#                                                                                    #
#                          FUNCIONES DE EDICION                                      #
# Funciones de edicion, transformacion, mapeo de LUTS y pre procesamiento para las   #
# imagenes                                                                           #
#                                                                                    #
######################################################################################

#--------------------------OPERACIONES DE UNA SOLA IMAGEN---------------------------------

#########################################################
#                 OPERADOR UMBRAL BINARIO               #
# Binariza la imagen                                    #
# @param: imagen, umbral                                #
#########################################################

def opUmbral(img, umbral):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i,j] < umbral):
                img[i,j] = 0
            else:
                img[i,j] = 255

#########################################################
#         OPERADOR INTERVALO UMBRAL ESCALA GRISES        #
# Ref: umbral es la intensidad (0,255)                   #
# Mantiene escala de grises en el intervalo umbral       #
# lo que este fuera del intervalo se transforma a @color #
# @param: imagen, umbral, color                          #
#########################################################

def opUmbralGrises(img, umbral, color):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (not(img[i,j] >= umbral[0] and img[i,j] <= umbral[1])):
                img[i,j] = color


#########################################################
#                 OPERADOR OFFSET                       #
# agrega/quita brillo a la imagen                       #
# @param: imagen, offset                                #
#########################################################

def opOffset(img, offset):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = img[i,j] + offset
            img[i,j] = max(img[i,j],0)
            img[i,j] = min(img[i,j],255)

##########################################################
#                                                        #
#           LUT: Transformacion lineal                   #
# Genera un vector con la transformacion lineal, donde   #
# el indice del vector es el valor de gris de la entrada #
# r (0-255) y s (0-255) es el resultado transformado     #
# a ese valor de intensidad r                            #
# //s = a*r + c                                          #
# @param: a, c                                           #
# @return: s                                             #
#                                                        #
##########################################################

def mapeo_ac(a, c):
    s = np.zeros(256)
    for i in range (s.shape[0]):
        s[i] = a*i + c
        s[i] = max(s[i], 0)
        s[i] = min(s[i], 255)
    return np.array(s)

def mapeo_negativo_ac(a,c):
    s = np.zeros(256)
    for i in range (s.shape[0]):
        s[i] = 255 - (a*i + c)
        s[i] = max(s[i], 0)
        s[i] = min(s[i], 255)
    return s


##########################################################
#                                                        #
#            LUT: Transformacion logaritmica             #
# Realiza un mapeo logaritmico en el vector s para la    #
# LUT                                                    #
# imagen de entrada con un rango dinámico grande,        #
# expande las intensidades oscuras y comprime las        #
# intensidades claras.                                   #
#                                                        #
# @param: c                                              #
# @return: s                                             #
#                                                        #
##########################################################

def mapeo_log(c):
    s = np.zeros(256)

    for i in range(256):
        s[i] = (c * np.log(1 + i)) / np.log(256) * 255
        s[i] = max(s[i],0)
        s[i] = min(s[i],255)
    s = s.astype(int)
    s = s.astype(np.uint8)
    return s


##########################################################
#                                                        #
#   LUT: Transformacion de potencia - Correccion gamma   #
# Realiza un mapeo potencial en el vector s para la LUT  #
# Imagen de entrada tiene un rango dinámico bajo, ex-    #
# pande las intensidades claras y comprime las intensi-  #
# dades oscuras.                                         #
#                                                        #
# @param: c                                              #
# @return: s                                             #
##########################################################

def mapeo_potencia(c,gamma):
    s = np.zeros(256)
    for i in range(256):
        s[i] = c * pow(i, gamma)
        s[i] = max(s[i], 0)
        s[i] = min(s[i], 255)
    s = s.astype(int)
    s = s.astype(np.uint8)
    return s

#---------------------------------OPERACIONES DE MULTIPLES IMAGENES----------------------------------------

##########################################################
#                                                        #
#               OPERADOR ARITMETICO                      #
# Realiza la op aritmetica de dos imagenes               #
# @param: img1, img2, tipo                               #
# @return: imgResultado                                  #
##########################################################

def opAritmeticas(img1, img2, tipo):
    TYPES = {
        "suma": cv.add(img1, img2),
        "resta": cv.subtract(img1, img2),
        "division": cv.divide(img1, img2),
        "multiplicacion": cv.multiply(img1, img2),
    }
    imgResultado = TYPES[tipo]
    return imgResultado


# @FALTA ARREGLAR
#########################################################
# Unir imagenes en una sola                             #
# Orientacion horizontal(a lo ancho) axis=0, vertical   #
# (a lo largo) axis=1                                   #
# @param: imgs[], axis                                  #
# @return: result                                       #
#########################################################

def joinImg(imgs, axis):
    max_x = 0
    max_y = 0
    for i in imgs:
        x=imgs[i].shape[0]
        y=imgs[i].shape[1]
        if(axis == 1):
            max_y += y
            if(max_x < x):
                max_x = x
        else:
            max_x += x
            if(max_y < y):
                max_y = y
    result = np.zeros((max_x, max_y, 3), dtype=imgs[i].dtype)
    #print("result shape", result.shape)
    x = 0
    y = 0
    for i in imgs:
        dx = x + imgs[i].shape[0]
        dy = y + imgs[i].shape[1]
        
        result[x:dx,y:dy,:] = imgs[i].copy()

        if(axis == 0):
            x = x + imgs[i].shape[0]
        else:
            y = y + imgs[i].shape[1]

    return result

