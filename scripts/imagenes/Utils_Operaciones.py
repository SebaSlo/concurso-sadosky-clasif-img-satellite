import cv2 as cv
import numpy as np

######################################################################################
#                                                                                    #
#                          FUNCIONES DE OPERACION                                    #
# Funciones que trabajan sobre las imagenes, pero no las alteran en si:              #
# Para obtener datos de las mismas, como por ejemplo aislar objetos u obtener ROIs   #
######################################################################################


##########################################################
#                                                        #
#               Aislar y ecuadrar objetos                #
#                                                        #
# Recorre por columna cada fila buscando el primer px    #
# de distinto @color al del fondo                        #
# @return: coordenadas de inicio y final del obj         #
##########################################################

def coord_bounding_box(img, color, margen):
    x_i = 0
    y_i = 0
    x_f = img.shape[1]
    y_f = 0
    flag = True
    #Busco inicio y final a lo ancho (x)
    for j in range(img.shape[1]):
        if(flag and np.average(img[:,j]) > 10):
            x_i = (j-margen if (j-margen)>0 else 0)
            flag = False
        if(not(flag) and np.average(img[:,j]) < 10):
            x_f = (j+margen if (j+margen)<img.shape[1] else img.shape[1])
            break
    
    #Busco inicio y final a lo largo (y)
    flag = True
    for i in range(img.shape[0]):
        if(flag and np.average(img[i,:]) > 10):
            y_i = (i-margen if (i-margen)>0 else 0)
            flag = False
        if(not(flag) and np.average(img[i,:]) < 10):
            y_f = (i+margen if (i+margen)>img.shape[0] else img.shape[0])
            break
        y_f = i

    return [(x_i, y_i), (x_f, y_f)]

def bounding_box(img, _color=(0,255,0)):
    margen = 4
    inicio, fin = coord_bounding_box(img, 0, margen)
    img_RGB = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.rectangle(img_RGB, inicio, fin, color=_color, thickness=1)

    return img_RGB, inicio, fin



######################################################################################
#                                                                                    #
#                          FUNCIONES DE OBTENCION DE DATOS                           #
#                                                                                    #
######################################################################################

##################################################
#                                                #
#             Busca el primer px de color        #
# Busca por fila - columna la primer ocurrencia  #
# de pixel.                                      #
# @return posicion de la mitad de la tapa        #
##################################################

def centroBotella(img):
    fila_tapa = 0
    pxs = 0
    flag = False
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] == 255):
                pxs += 1
                fin_tapa = j
                flag = True
        if (flag):
            fila_tapa = i
            break
    col_tapa = fin_tapa-int(pxs/2)
    pt_med = (fila_tapa,col_tapa)
    return np.array(pt_med)

##########################################################
#Cuenta los px de un determinado color en linea vertical #
# u horizontal                                           #
# default blanco, vertical                               #
##########################################################

def contadorPxColor(img, pt, color=255, vertical=True):
    cont = 0
    while (img[pt[0],pt[1]] == color):
        cont += 1
        if(vertical):
            pt[0] += 1
        else:
            pt[1] += 1
    return cont






