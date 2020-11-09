import numpy as np
import math
import cv2 as cv
from TP1_utils import *
from matplotlib import pyplot as plt

##########################################################
#                                                        #
#               MASCARA BINARIA                          #
# Crea una mascara binaria a partir de un umral          #
# @param: img, umbral                                    #
# @return: mascara                                       #
##########################################################

def mascara_binaria(img, umbral):
    """ NORMALIZAR A 255 y pasar a uint8"""
    mascara = img.copy()
    s = mapeo_binario(umbral)
    mascara = cv.LUT(mascara,s)
    return mascara

##########################################################
#                                                        #
#           LUT: Transformacion binaria                  #
# Genera un vector para una transformacion binaria       #
# @param: umbral                                         #
# @return: s                                             #
#                                                        #
##########################################################

def mapeo_binario(umbral):
    lim = 256 - umbral
    s = np.zeros(umbral)
    s = np.hstack((s,np.ones(lim)))
    return s


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
    #s = np.array(s)
    s = s.astype(int)
    s = s.astype(np.uint8)
    return s

def mapeo_negativo_ac(a,c):
    s = np.zeros(256)
    for i in range (s.shape[0]):
        s[i] = 255 - (a*i + c)
        s[i] = max(s[i], 0)
        s[i] = min(s[i], 255)
    s = s.astype(int)
    s = s.astype(np.uint8)
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
    """expande las claras gamma exponente"""
    s = np.zeros(256)
    for i in range(256):
        s[i] = c * pow(i, gamma)
        s[i] = max(s[i], 0)
        s[i] = min(s[i], 255)
    s = s.astype(np.uint8)
    return s

#testear
def mapeo_hue(parametro):
    s = np.zeros(256)
    for i in range(256):
        s[i] = (i + parametro) % 180
    s = s.astype(int)
    s = s.astype(np.uint8)
    return s

def umbralBinario(img,p1,p2):
    tablaLut1=np.array(range(0,256))
    tablaLut1[0:p1]=0
    tablaLut1[p1:p2]=255
    tablaLut1[p2:255]=0
    tablaLut1=tablaLut1.astype(np.uint8)
    res=cv.LUT(img,tablaLut1)
    return res

##########################################################
#                                                        #
#               OPERADOR ARITMETICO                      #
# Suma dos imagenes                                      #
# @param: img1, img2                                     #
# @return: imgResultado                                  #
##########################################################

def opAritmeticas(img1, img2, tipo):
    TYPES = {
        "suma": cv.add(img1, img2),
        "resta": cv.subtract(img1, img2),
        "division": cv.divide(img1, img2),
        "multiplicacion": cv.multiply(img1, img2),
    }

    return TYPES[tipo]

###########################################################
#                                                         #
#          OPERADOR ARTIMETICO PROMEDIO DE IMAGENES       #
# Sirve para eliminar ruido de imagenes (ruido gaussiano) #
# Eliminar objetos en movimiento de la imagen (OBTENER    #
# FONDO)                                                  #
# Recibe N imagenes y realiza su promedio                 #
# @param: imgs[]                                          #
# @return: promImg                                        #
###########################################################

def suma_promediada(imgs):
    N = len(imgs)
    imgResultado = np.array(imgs)
    imgResultado = imgResultado.sum(axis=0)
    imgResultado = imgResultado/N
    imgResultado.astype(int)
    imgResultado = imgResultado.astype(np.uint8)
    return imgResultado




#################################################################
#                                                               #
#         Variacion de valores mediante un trackbar             #
#                                                               #
# Con un trackbar previamente inicializado en una image         #
# Recibe un valor desde el cual arranca el parametro y se le    #
# un delta a ese valor que esta dado por la posicion del        #
# trackbar sobre un "delta" que da el factor de escala (1, 0.1  #
# 0.01, etc)                                                    #
# Si el parametro admite valores negativos, entonces el track-  #
# bar debera ser inicializado en 50                             #
#                                                               #
# @param: trackbarName, windowName, valor_inicial, delta        #
# @return: val                                                  #                                        
#################################################################

def param_trackBar(trackbarName, windowName, valor_inicial=0, delta=1, negativo=False):
    pos = cv.getTrackbarPos(trackbarName, windowName)
    if(negativo):
        pos = cv.getTrackbarPos(trackbarName, windowName) - 50
    val = valor_inicial + (pos * delta)
    return val

# cv.createTrackBar se le debe pasar una funcion como parametro, como
# se necesita solo la posicion del track bar, le pasamos una que no
# hace nada
def track_change(val):
    pass

#################################################################
#                                                               #
#                  CAPTURA PUNTOS EN LA IMAGEN                  #
#                                                               #
#################################################################

def capturar_punto(event, x, y, flags, param):
    global mouseX,mouseY, imgn
    if event == cv.EVENT_LBUTTONDOWN:
            print ("posicion elegida: (",x,",",y,"), presione 'a' o 'c' para confirmar.")
            mouseX,mouseY = x,y

def elegir_punto(image):
    global imgn
    imgn = image
    cv.namedWindow("image")
    cv.setMouseCallback("image", capturar_punto)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv.imshow("image", imgn)
        key = cv.waitKey(20) & 0xFF

        if key == 27:
            break
        elif key == ord('a'):
            return [mouseX, mouseY]
        elif key == ord('c'):
            return [mouseX, mouseY]

#####################################################
#                                                   #
#               RESIZE IMG                          #
# A partir de una imagen se la reescala hacia abajo #
# en un porcentaje                                  #
# @param: img, porcentaje                           #
# @return: resized                                  #
#####################################################

def resize_img(img, porcentaje):
    width = int(img.shape[1] * porcentaje / 100)
    height = int(img.shape[0] * porcentaje / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resized 

#####################################################
#                                                   #
#        Dibuja el histograma de la imagen          #
#                                                   #
#####################################################
def histograma(img, title="Histograma"):
    plt.figure()
    plt.hist(img.flatten(), 255)
    plt.title(title)

def graficar(img,maximo,minimo,mapa,titulo=''):
    ventana = plt.figure()
    # ventana.canvas.set_window_title(titulo)
    plt.axis("off")
    plt.imshow(img, vmax=maximo, vmin=minimo, cmap=mapa)
    plt.title(titulo)


#####################################################
#                                                   #
#                    COLOR                          #
#                                                   #
#####################################################

#####################################################
#                                                   #
#             Extraer I de img BGR                  #
#                                                   #
# @param: img (formato BGR)                         #
# @return: I (componente de intensidad)             #
#####################################################

def extraerI_BGR(img):
    I = np.array(img.shape)
    I = I.astype(np.uint16)
    [B,G,R] = cv.split(img)
    I= (B + G + R)/3
    I = I.astype(np.uint8)
    return I

#####################################################
#              ARREGLAR/VERIFICAR                   #
#             Transforma de HSI a BGR               #
#                                                   #
# @param: img (formato HSI)                         #
# @return: img_bgr (img formato BGR)                #
#####################################################
def HSItoBGR(img_hsi):
    #img_hsi = img_hsi.astype(np.uint16)
    [H,S,I] = cv.split(img_hsi)
    H = 360*(H/179)
    S = S/255
    I = I/255

    #Creo los canales B G R con el mismo tamaño q H S I
    B = np.zeros(H.shape)
    G = np.zeros(H.shape)
    R = np.zeros(H.shape)
    
    for i in range (H.shape[0]):
        for j in range (H.shape[1]):
            if ((H[i,j]>=0) & (H[i,j]<120)):
                B[i,j] = I[i,j] * (1 - S[i,j])
                R[i,j] = I[i,j] * (1 + (S[i,j]*math.cos(H[i,j])/(math.cos(60 - H[i,j]))))
                G[i,j] = 3 * I[i,j] - (R[i,j] + B[i,j])
            
            if ((H[i,j]>=120) & (H[i,j]<240)):
                H[i,j] = H[i,j] - 120
                
                R[i,j] = I[i,j]*(1 - S[i,j])
                G[i,j] = I[i,j] * (1 + (S[i,j]*math.cos(H[i,j])/(math.cos(60 - H[i,j]))))
                B[i,j] = 3 * I[i,j] - (R[i,j] + G[i,j])
            
            if ((H[i,j]>=240) & (H[i,j]<=360)):
                H[i,j] = H[i,j] - 240

                G[i,j] = I[i,j]*(1 - S[i,j])
                B[i,j] = I[i,j] * (1 + (S[i,j]*math.cos(H[i,j])/(math.cos(60 - H[i,j]))))
                R[i,j] = 3 * I[i,j] - (G[i,j] + B[i,j])

    img_bgr = cv.merge([B,G,R])
    img_bgr = img_bgr.astype(np.uint8)
    return img_bgr



