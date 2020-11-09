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