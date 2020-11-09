import cv2 as cv
import numpy as np

######################################################################################
#                                                                                    #
#                          FUNCIONES DE EDICION                                      #
#                                                                                    #
######################################################################################

#Unir imagenes en una sola
#Orientacion horizontal(a lo ancho) axis=0, vertical(a lo largo) axis=1
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

#########################################################
#                                                       #
#               TRANSFORMACIONES LINEALES               #
#                                                       #
#########################################################


#########################################################
#                 OPERADOR UMBRAL BINARIO               #
# Binariza la imagen                                    #
# @param: imagen, umbral                                #
#########################################################

def opUmbral(img, umbral):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
           img[i,j] = 0 if img[i,j]<umbral else 255

#########################################################
#         OPERADOR INTERVALO UMBRAL ESCALA GRISES        #
# Ref: umbral es la intensidad (0,255)                   #
# Mantiene escala de grises en el intervalo              #
# lo que este fuera del @intervalo se transforma a       #
# @intensidad                                            #
#                                                        #
# @param: imagen, intervalo, color                       #
##########################################################

def opUmbralGrises(img, intervalo, color):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (not(img[i,j] >= intervalo[0] and img[i,j] <= intervalo[1])):
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
            img[i,j] = max(img[i,j], 0)
            img[i,j] = min(img[i,j], 255)
            


######################################################################################
#                                                                                    #
#                          FUNCIONES OPERATIVAS                                      #
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




