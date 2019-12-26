import os
ruta = os.getcwd() + '/'
src = os.path.abspath(os.path.join(ruta , '..'))
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
stn, scn, met, calib_orig, start, OF = conf[0], conf[1], conf[2], conf[3], conf[4], conf[5]
del conf

# Importar archivo de calibración de partida
calib = pd.read_csv(ruta + '/'  + calib_orig, header=None)
idx = ['FC' + str(i) for i in range(1, 10)] + ['FC0', 'Bnieve', 'DDF1', 'DDF2', 'Tb', 'Bppt', 'USLE1', 'USLE2', 'USLE3', 'p1', 'H3max']
calib.index = idx
calib = calib[0]
# convertir en NaN los parametros a calibrar
par_cal = ['FC3', 'FC4', 'FC5', 'FC6', 'FC7', 'FC8', 'FC9', 'H3max']
calib[par_cal] = np.NaN

############
# MÉTODO 1 #
############
if met == 'met1':
    print('############')
    print('# MÉTODO 1 #')
    print('############')
    
    # Paso 1. Crear archivos del modelo
    # ---------------------------------
    crear_tet(ruta, stn, scn, met, flow='quick')
    crear_tet(ruta, stn, scn, met, flow='slow')
    crear_tet(ruta, stn, scn, met, flow='total')
    
    # Paso 2. Calibrar el caudal total
    # --------------------------------
    print('CALIBRACIÓN DEL CAUDAL TOTAL')
    print('----------------------------')
    # Definir parámetros del modelo
    crear_calib(calib, ruta, 'total', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'total', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_total_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración automática
    lanzar_SCEUA(ruta, 'total', met, scn)
    try:
        calib_total = lanzar_SCEUA.calib
        obs_total = lanzar_SCEUA.obs
        sim_total = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')
        
    # Paso 3. Simular los caudales desagregados
    # -----------------------------------------
    # Caudal rápido
    #calib_quick = calib_total.copy()
    #calib_quick['FC8'] = np.nan
    #crear_calib(calib_quick, ruta, 'quick', met)
    

        
############
# MÉTODO 2 #
############        
elif met == 'met2':
    print('############')
    print('# MÉTODO 2 #')
    print('############')
    
    # Paso 1. Crear archivos del modelo
    # ---------------------------------
    crear_tet(ruta, stn, scn, met, flow='quick')
    crear_tet(ruta, stn, scn, met, flow='slow')
    crear_tet(ruta, stn, scn, met, flow='total')

    # Paso 2. Calibrar el caudal rápido
    # ---------------------------------
    print('CALIBRACIÓN DEL CAUDAL RÁPIDO')
    print('-----------------------------')
    # Definir parámetros del modelo
    crear_calib(calib, ruta, 'quick', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'quick', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_quick_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración automática
    lanzar_SCEUA(ruta, 'quick', met, scn)
    try:
        calib_quick = lanzar_SCEUA.calib
        obs_quick = lanzar_SCEUA.obs
        sim_quick = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')
    print()
        
    # Paso 3. Calibrar el caudal lento
    # --------------------------------
    print('CALIBRACIÓN DEL CAUDAL LENTO')
    print('----------------------------')
    # Definir parámetros del modelo
    calib_slow = calib_quick.copy()
    calib_slow['FC8'] = np.nan # para dar el valor por defecto a 'FC8'
    crear_calib(calib_slow, ruta, 'slow', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'slow', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_slow_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración automática
    lanzar_SCEUA(ruta, 'slow', met, scn)
    try:
        calib_slow = lanzar_SCEUA.calib
        obs_slow = lanzar_SCEUA.obs
        sim_slow = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')
    print()

    # Paso 4. Calibracion caudal total
    # --------------------------------
    print('CALIBRACIÓN DEL CAUDAL TOTAL')
    print('----------------------------')
    # Definir parámetros del modelo
    calib_total = calib_quick.copy()
    calib_total[['FC7', 'FC8']] = calib_slow[['FC7', 'FC8']]
    crear_calib(calib_total, ruta, 'total', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'total', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_total_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración automática
    lanzar_SCEUA(ruta, 'total', met, scn)
    try:
        calib = lanzar_SCEUA.calib
        obs_total = lanzar_SCEUA.obs
        sim_total = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')


############
# MÉTODO 3 #
############        
elif met == 'met3':
    print('############')
    print('# MÉTODO 3 #')
    print('############')
    
    # Paso 1. Crear archivos del modelo
    # ---------------------------------
    crear_tet(ruta, stn, scn, met, flow='quick')
    crear_tet(ruta, stn, scn, met, flow='slow')
    crear_tet(ruta, stn, scn, met, flow='total')

    # Paso 2. Calibrar el caudal lento
    # --------------------------------
    print('CALIBRACIÓN DEL CAUDAL LENTO')
    print('----------------------------')
    # Definir parámetros del modelo
    crear_calib(calib, ruta, 'slow', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'slow', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_slow_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración automática
    lanzar_SCEUA(ruta, 'slow', met, scn)
    try:
        calib_slow = lanzar_SCEUA.calib
        obs_slow = lanzar_SCEUA.obs
        sim_slow = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')
    print()
        
    # Paso 3. Calibrar el caudal rápido
    # --------------------------------
    print('CALIBRACIÓN DEL CAUDAL RÁPIDO')
    print('-----------------------------')
    # Definir parámetros del modelo
    calib_quick = calib_slow.copy()
    calib_quick[['FC4', 'FC6']] = np.nan # para dar el valor por defecto a 'FC4' y 'FC6'
    crear_calib(calib_quick, ruta, 'quick', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'quick', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_quick_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración automática
    lanzar_SCEUA(ruta, 'quick', met, scn)
    try:
        calib_quick = lanzar_SCEUA.calib
        obs_quick = lanzar_SCEUA.obs
        sim_quick = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')
    print()

    # Paso 4. Calibracion caudal total
    # --------------------------------
    print('CALIBRACIÓN DEL CAUDAL TOTAL')
    print('----------------------------')
    # Definir parámetros del modelo
    calib_total = calib_quick.copy()
    calib_total[['FC7', 'FC8']] = calib_slow[['FC7', 'FC8']]
    crear_calib(calib_total, ruta, 'total', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'total', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_total_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración automática
    lanzar_SCEUA(ruta, 'total', met, scn)
    try:
        calib = lanzar_SCEUA.calib
        obs_total = lanzar_SCEUA.obs
        sim_total = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')

        
############
# MÉTODO 4 #
############        
elif met == 'met4':
    print('############')
    print('# MÉTODO 4 #')
    print('############')
    
    # Paso 1. Crear archivos del modelo
    # ---------------------------------
    crear_tet(ruta, stn, scn, met, flow='quick')
    crear_tet(ruta, stn, scn, met, flow='total')

    # Paso 2. Calibrar el caudal rápido
    # ---------------------------------
    print('CALIBRACIÓN DEL CAUDAL RÁPIDO')
    print('-----------------------------')
    # Definir parámetros del modelo
    crear_calib(calib, ruta, 'quick', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'quick', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_quick_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración automática
    lanzar_SCEUA(ruta, 'quick', met, scn)
    try:
        calib = lanzar_SCEUA.calib
        obs_quick = lanzar_SCEUA.obs
        sim_quick = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')
    print()

    # Paso 3. Calibracion caudal total
    # --------------------------------
    print('CALIBRACIÓN DEL CAUDAL TOTAL')
    print('----------------------------')
    # Crear archivo de calibración
    calib['FC8'] = np.nan # para dar el valor por defecto a 'FC8'
    crear_calib(calib, ruta, 'total', met)
    # Crear archivo de configuración de la calibración automática
    #crear_VarSCEUA(par_cal, ruta, 'total', met, start=start, OF=OF)
    varfile = 'Var-SCEUA_total_' + met + '.txt'
    copyfile(src + '/' + varfile, ruta + '/' + varfile)
    # Lanzar la calibración
    lanzar_SCEUA(ruta, 'total', met, scn)
    try:
        calib = lanzar_SCEUA.calib
        obs_total = lanzar_SCEUA.obs
        sim_total = lanzar_SCEUA.sim
    except:
        print('No fue posible simular')

