import numpy as np
import pandas as pd
import os
from shutil import copyfile

os.chdir('F:\\Codigo\\Python\\TETIS')
from funciones_exportar_TETIS import export_heading, export_series, export_control
from funciones_resultados_TETIS import leer_caudal, leer_SCEUA, NSE, sesgo, RMSE, KGE


def crear_calib(FCs, path, flow=None, met=None):
    """Crea el archivo de calibración ('Calib.txt') del modelo TETIS.
    
    Entradas:
    ---------
    FCs:     series. Contiene los valores de los parámetros del archivo 'Calib.txt' distintos del valor por defecto
    path:    string. Carpeta en la que guardar el archivo
    flow:    string. Tipo de caudal al que se refieren los valores de la calibración. Sólo influye en el nombre del archivo exportado
    met:     string. Método de calibración empleado en la optimización de los parámetros. Sólo influye en el nombre del arcihvo exportado
    
    Salidas:
    --------
    Archivo 'Calib_flow_met.txt'"""
    
    # Valores de 'Calib.txt' por defecto
    values = [1, 1, 0.2, 1, 0.4, 10, 1, 200, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 100000]
    idx = ['FC' + str(i) for i in range(1, 10)] + ['FC0', 'Bnieve', 'DDF1', 'DDF2', 'Tb', 'Bppt', 'USLE1', 'USLE2', 'USLE3', 'p1', 'H3max']
    FCs_def = pd.Series(data=values, index=idx)
    
    # identificar parametros nulos en funcion del tipo de caudal
    if flow == 'quick':
        nulls = ['FC8']
    elif flow == 'slow':
        nulls = ['FC4', 'FC6']
    else:
        nulls = []
        
    # completar/corregir los parametros de entrada
    for FC in FCs.index:
        if FC in nulls:
            FCs[FC] = 0
        elif np.isnan(FCs[FC]):
            FCs[FC] = FCs_def[FC]
    
    # crear archivo Calib.txt
    filename = 'Calib_' + flow + '_' + met + '.txt'
    with open(path + filename, 'w+') as f:
        for value in FCs:
            f.write(str(round(value, 7)) + '\n')


def crear_VarSCEUA(par_cal, ruta, flow, met,
                   start=1, OF=6, p_forma=2.0, n_dias=30, A_pond=0):
    """Crea el archivo de configuración de la calibración automática del módulo hidrológico de TETIS.
    
    Entradas:
    ---------
    par_cal: list. Nombre de los parámetros a calibrar
    ruta:    string. Carpeta en la que guardar el archivo
    flow:    string. Tipo de caudal al que se refieren los valores de la calibración. Sólo influye en el nombre del archivo exportado
    met:     string. Método de calibración empleado en la optimización de los parámetros. Sólo influye en el nombre del arcihvo exportado
    start:   integer. Día de la simulación a partir del que calcular la función objetivo
    OF:      integer. Función objetivo a emplear: 4-RMSE, 5-volumen acumulado, 6-NSE
    p_forma: float. Valor del parámetro de forma para el cálculo de la función objetivo HMLE
    n_dias:  integer. Número de días en los que agrupar el caudal para el cálculo de la función objetivo RMSE mensual
    A_pond:  integer. Si ponderar (1) o no (0) la función objetivo según el área de la cuenca de los puntos aforados
    
    Salidas:
    --------
    Archivo 'Var-SCEUA_flow_met.txt'"""
    
    # TABLA POR DEFECTO DE RANGOS Y VALOR INICIAL DE LOS PARÁMETROS
    # -------------------------------------------------------------
    # definir valores mínimos, máximos y por defecto
    mins = [0.01, 0, 0, 0.001, 0, 0.001, 0, 0.001, 0, 0, 0, 0, 0, 0, -0.2, 0, 0, 1, 3, -2, 0, 0]
    maxs = [3, 2, 1.5, 10, 1.5, 5000, 1, 50000, 1.5, 100, 100, 100, 100, 100, 0.2, 1, 1, 3, 6, 4, 3, 1000000]
    defs = [1, 1, 0.2, 1, 0.4, 10, 0.5, 200, 1, 50, 0.5, 0.5, 0.5, 50, 0, 1, 1, 1, 1, 1, 0, 100000]
    
    # definir el nombre de los parámetros calibrables
    # factores correctores
    idx1 = ['FC' + str(i) for i in range(1, 10)]
    # estados iniciales
    idx2 = ['H' + str(i) for i in range(1, 6)]
    # otras variables
    idx3 = ['Bppt', 'Bnieve', 'FC0', 'DDF1', 'DDF2', 'Tb', 'p1', 'H3max']
    idx = idx1 + idx2 + idx3
    
    # generar data frame
    VarSCEUA = pd.DataFrame(data=[mins, maxs, defs], columns=idx).T
    VarSCEUA.columns = ['min', 'max', 'init']
    VarSCEUA['cal'] = 'F'
    VarSCEUA.loc[par_cal, 'cal'] = 'T'
    
    # EXPORTAR ARCHIVO 'Var-SCEUA.txt'
    # --------------------------------
    filename = 'Var-SCEUA_' + flow + '_' + met + '.txt'
    with open(ruta + filename, 'w+') as f:
        # Exportar variables de cálculo de la función objetivo
        f.write(str(start) + '\n')
        f.write(str(OF) + '\n')
        f.write(str(round(p_forma, 2)) + '\n')
        f.write(str(n_dias) + '\n')
        if A_pond not in [0, 1]:
            print('ERROR. "A_pond" debe ser 0 ó 1')
            return
        else:
            f.write(str(A_pond) + '\n')
        
        # Exportar la configuración del espacio de parámetros a analizar
        for FC in VarSCEUA.index:
            val = VarSCEUA.loc[FC,:]
            f.write('   {0:7.5f}   {1:7.5f}   {2:7.5f}      {3:1}\n'.format(val[0], val[1], val[2], val[3]))


def crear_tet(ruta, nombre, scn, met=None, flow=None):
    """Generador del archico .tet de un modelo TETIS.
    
    Entradas:
    ---------
    ruta:    string. Ruta donde se guardará el archivo .tet
    nombre:  string. Nombre (cuenca hidrográfica, estación) que aparecerá al inicio del
                     nombre del archivo .tet
    scn:     string. Escenario
    met:     string. Método de calibración secuencial
    flow:    string. Tipo de caudal: 'total', 'quick', 'slow'
    
    Salida:
    -------
    En la carpeta 'ruta' se genera el archivo 'nombre flow_met_scn.tet'"""
    
    veg, soil = scn[:-4], scn[-3:]
    modelname = nombre + ' ' + flow + '_' + met + '_' + scn + '.tet'
    
    # exportar archivo .tet
    with open(ruta + modelname, 'w+') as file:
        # write attributes
        file.write(ruta +'\n')
        file.write('Paramgeo.txt\n')
        file.write('Calib_' + flow + '_' + met + '.txt\n')
        file.write('Topolco_' + scn + '.sds\n')
        file.write('Hantec2008.sds\n')
        file.write('2008-2014_' + flow + '_' + scn + '_p500.txt\n')
        file.write('FactorETmes_PdE.txt\n')
        file.write('CurvasHV.txt\n')
        file.write('Hantec2014ts.sds\n')
        file.write('Q' + flow + '_' + met + '_' + scn + '.res\n')
        file.write('Nieve2008.asc\n')
        file.write('Nieve2014.asc\n')
        file.write('dem.asc\n')
        file.write('Hu_' + scn + '.asc\n')
        file.write('K' + soil + '.asc\n')
        file.write('K' + soil + '.asc\n')
        file.write('cobveg_' + veg + '.asc\n')
        file.write('slope.asc\n')
        file.write('fdir.asc\n')
        file.write('Acum.asc\n')
        file.write('Riego.asc\n')
        file.write('Control.txt\n')
        file.write('Riego.txt\n')
        file.write('Recorta.txt\n')
        file.write('RegHomog.asc\n')
        file.write('Pptacum.asc\n')
        file.write('Var-SCEUA_' + flow + '_' + met + '.txt\n')
        file.write('Res-SCEUA_' + flow + '_' + met + '.txt\n')
        file.write('Mapafcs.asc\n')
        file.write('OrdenRiego.asc\n')
        file.write('K' + soil + '.asc\n')
        file.write('K' + soil + '.asc\n')
        file.write('K' + soil + '.asc\n')
        file.write('vel.asc\n')
        file.write('Settings.txt\n')
        file.write('rad01.asc\n')
        file.write('karst.asc\n')
        file.write('manantiales.txt\n')
        file.write('TipoRiego.txt\n')
        file.write('Acuifero.asc\n')
        file.write('FactvegCultivos.txt\n')
        file.write('CalibVeg.txt\n')
        file.write('cero.asc\n')
    file.close()


def lanzar_SCEUA(path, flow, met, scn=None):
    """Lanza la calibración automática del modelo TETIS según la secuencia de calibración del método 'met'.
    
    Entradas:
    ---------
    FCs:     series. Contiene los valores de los parámetros del archivo 'Calib.txt' distintos del valor por defecto
    path:    string. Carpeta donde se encuentra el modelo TETIS y todos sus archivos auxiliares
    flow:    string. Tipo de caudal al que se refieren los valores de la calibración. Sólo influye en el nombre de algunos archivos
    met:     string. Método de calibración secuencial empleado en la optimización de los parámetros. Sólo influye en el nombre de algunos archivos
    scn:     string. Escenario de la simulación. Sólo influye en el nombre de algunos archivos
    
    Salidas:
    --------
    Como métodos:
    calib:   series. Valores calibrados de los parámetros del modelo
    obs:     series. Serie de caudal observado
    sim:     series. Serie de caudal simulado con los parámetros optimizados"""
    
    # copiar archivo .tet como 'FileSSP.txt'
    os.chdir(path)
    for file in os.listdir():
        if ('tet' in file[-3:]) and (met in file) and (flow in file):
            break
    copyfile(file, 'FileSSP.txt')
    
    # generar el archivo TOPOLCO.sds si no existiera
    topolco = 'Topolco_' + scn + '.sds'
    if topolco not in os.listdir():
        print('Generando archivo:', topolco)
        code = os.system('Toparc.exe')
        if code != 0:
            print('ERROR. No se generó el archivo TOPOLCO.')
            
    # calibración automática
    print('Calibrando modelo:', file)
    code = os.system('CalibAutom.exe')
    if code != 0:
        print('ERROR. La calibración no fue correcta.')
        return
    else:
        # importar los valores calibrados de los parametros
        leer_SCEUA(path, 'Res-SCEUA_' + flow + '_' + met + '.txt')
        pars = leer_SCEUA.parametros
        print('NSE =', leer_SCEUA.resultados.Nash.max())
        
        # importar los archivos 'Calib.txt' originales
        calib = pd.read_csv(path + 'Calib_' + flow + '_' + met + '.txt', header=None)
        calib.index = ['FC' + str(i) for i in range(1, 10)] + ['FC0', 'Bnieve', 'DDF1', 'DDF2', 'Tb', 'Bppt', 'USLE1', 'USLE2', 'USLE3', 'p1', 'H3max']
        calib = calib[0]
        
        # modificar los valores de los parametros calibrados
        for FC in pars.index:
            if FC[0] == 'R':
                par_name = 'FC' + FC[1]
            elif FC[:3] == 'Cap':
                par_name = 'H3max'
            calib[par_name] = pars.loc[FC, 'optimo']

        # exportar el archivo 'Calib_flow.txt' modificado
        crear_calib(calib, path, flow, met)
        lanzar_SCEUA.calib = calib
        
        # simular
        print('Simulando modelo:', file)
        code = os.system('Tetis.exe')
        if code != 0:
            print('ERROR. La simulación no fue correcta.')
            return
        else:
            leer_caudal(path, 'Q' + flow + '_' + met + '_' + scn + '.res',
                        observed=True, plot=False, export=False)
            lanzar_SCEUA.obs = leer_caudal.observado
            lanzar_SCEUA.sim = leer_caudal.simulado
            nse = NSE(leer_caudal.observado, leer_caudal.simulado)
            print('NSE =', round(nse, 3))
            
    os.remove('FileSSP.txt') 

