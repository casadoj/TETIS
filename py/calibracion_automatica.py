import os
ruta = os.getcwd() + '/'
import numpy as np
import pandas as pd
import sys
from shutil import copyfile
sys.path.insert(0, 'F:\\Codigo\\Python\\TETIS')
from funciones_resultados_TETIS import leer_caudal, leer_SCEUA, NSE, sesgo, RMSE, KGE
from funciones_calibracion_TETIS import crear_tet, crear_calib, lanzar_SCEUA

# leer el archivo de configuración
os.chdir(ruta)
with open(ruta + 'Config.txt', 'r') as config:
    conf = config.readlines()
config.close()
conf = [line.split()[0] for line in conf]
stn, scn, met = conf[0], conf[1], conf[2]
del conf


# Paso 1. Crear archivos del modelo
if met == 'met1':
    crear_tet(ruta, stn, scn, met, flow='total')
elif met in ['met2', 'met3', 'met4']:
    crear_tet(ruta, stn, scn, met, flow='quick')
    if met in ['met2', 'met3']:
        crear_tet(ruta, stn, scn, met, flow='slow')
    crear_tet(ruta, stn, scn, met, flow='total')


# Paso 2. Calibrar el caudal rápido
# Crear archivo de calibración
filename = ruta + '\Calib_met0_median_quick.txt'
calib = pd.read_csv(filename, header=None)
idx = ['FC' + str(i) for i in range(1, 10)] + ['FC0', 'Bnieve', 'DDF1', 'DDF2', 'Tb', 'Bppt', 'USLE1', 'USLE2', 'USLE3', 'p1', 'H3max']
calib.index = idx
calib = calib[0]
# convertir en NaN los parametros a calibrar
par_cal = ['FC3', 'FC4', 'FC5', 'FC6', 'FC7', 'FC8', 'FC9', 'H3max']
calib[par_cal] = np.NaN
# exportar calib
crear_calib(calib, ruta, 'quick', met)


# Lanzar la calibración
lanzar_SCEUA(ruta, 'quick', met, scn)
try:
    calib = lanzar_SCEUA.calib
    obs_quick = lanzar_SCEUA.obs
    sim_quick = lanzar_SCEUA.sim
except:
    print('No fue posible simular')


# Paso 2. Calibracion caudal total
# Crear archivo de calibración
calib['FC8'] = np.nan # para dar el valor por defecto a 'FC8'
crear_calib(calib, ruta, 'total', met)

# Lanzar la calibración
lanzar_SCEUA(ruta, 'total', met, scn)
try:
    calib = calibracion_automatica.calib
    obs_total = calibracion_automatica.obs
    sim_total = calibracion_automatica.sim
except:
    print('No fue posible simular')

