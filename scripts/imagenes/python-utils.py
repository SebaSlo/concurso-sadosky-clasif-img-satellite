import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

######################################################################################
#                                                                                    #
#                          FUNCIONES DE EDICION                                      #
#                                                                                    #
######################################################################################



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

#################################################################
#                                                               #
#                   FUNCIONES DE GUI                            #
# Funciones que operan sobre la interfaz de usuario y pueden    #
# interactuar con los scripts                                   #
# ###############################################################

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

def param_trackBar(trackbarName, windowName, valor_inicial, delta, negativo=False):
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