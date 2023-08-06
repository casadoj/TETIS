import numpy as np
import pandas as pd
import os
from shutil import copyfile
from datetime import datetime, timedelta
import copy

# os.chdir('F:\\Codigo\\Python\\TETIS')
from funciones_exportar_TETIS import export_heading, export_series, export_control
#from funciones_resultados_TETIS import leer_resultados, leer_SCEUANSE, sesgo, RMSE, KGE


def lanzar_topolco(ruta, tet, rutaTETIS='C:/ProgramFiles/Tetis9/', verbose=True):
    """Lanza una simulación de TETIS.
    
    Entradas:
    ---------
    ruta:      string. Ruta donde se encuentra el proyecto TETIS
    tet:       string. Nombre del archivo de proyecto (.tet)
    rutaTETIS: string. Carpeta donde está instalado TETIS
    verbose:   boolean. Si se quiere mostrar el proceso por pantalla
    """
    
    rutaOrig = os.getcwd()
    os.chdir(ruta)
    
    # leer archivo de proyecto
    with open(tet) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    f.close()
    # nombre del archivo topolco a generar
    topolco = lines[3]
    
    # copiar archivo de ejecución de TETIS
    copyfile(rutaTETIS + 'bin/Toparc.exe', 'Toparc.exe')
    
    # hacer una copia del archivo .tet
    copyfile(tet, 'FileSSP.txt')
    
    # lanzar la simulación
    if verbose:
        print('Generando archivo {0}{1}'.format(ruta, topolco))
    code = os.system('Toparc.exe')
    if code != 0:
        print('ERROR. No se generaron los puntos de control.')
        
    # eliminar archivos temporales
    os.remove('Toparc.exe')
    os.remove('FileSSP.txt') 
    os.chdir(rutaOrig)

    
def lanzar_control(ruta, tet, rutaTETIS='C:/ProgramFiles/Tetis9/', verbose=True):
    """Lanza una simulación de TETIS.
    
    Entradas:
    ---------
    ruta:      string. Ruta donde se encuentra el proyecto TETIS
    tet:       string. Nombre del archivo de proyecto (.tet)
    rutaTETIS: string. Carpeta donde está instalado TETIS
    verbose:   boolean. Si se quiere mostrar el proceso por pantalla
    """
    
    rutaOrig = os.getcwd()
    os.chdir(ruta)
    
    # leer archivo de proyecto
    with open(tet) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    f.close()
    # nombre del archivo control a generar
    control = lines[21]
    
    # copiar archivo de ejecución de TETIS
    copyfile(rutaTETIS + 'bin/Control.exe', 'Control.exe')
    
    # hacer una copia del archivo .tet
    copyfile(tet, 'FileSSP.txt')
    
    # lanzar la simulación
    if verbose:
        print('Generando archivo {0}{1}'.format(ruta, control))
    code = os.system('Control.exe')
    if code != 0:
        print('ERROR. No se generaron los puntos de control.')
        
    # eliminar archivos temporales
    os.remove('Control.exe')
    os.remove('FileSSP.txt')    
    os.chdir(rutaOrig)
    
    
def lanzar_simulacion(ruta, tet, rutaTETIS='C:/ProgramFiles/Tetis9/', verbose=True):
    """Lanza una simulación de TETIS.
    
    Entradas:
    ---------
    ruta:      string. Ruta donde se encuentra el proyecto TETIS
    tet:       string. Nombre del archivo de proyecto (.tet)
    rutaTETIS: string. Carpeta donde está instalado TETIS
    """
    
    rutaOrig = os.getcwd()
    os.chdir(ruta)
    
    # copiar archivo de ejecución de TETIS
    copyfile(rutaTETIS + 'bin/Tetis.exe', 'Tetis.exe')
    
    # hacer una copia del archivo .tet
    copyfile(tet, 'FileSSP.txt')
    
    # lanzar la simulación
    if verbose:
        print('Simulando {0}{1}'.format(ruta, tet))
    code = os.system('Tetis.exe')
    if code != 0:
        print('ERROR. La simulación no funcionó.')
        
    # eliminar archivos temporales
    os.remove('Tetis.exe')
    os.remove('FileSSP.txt')     
    os.chdir(rutaOrig)


def crear_Hantec(ruta, tet, H1=0, H2=0, H3=0, H4=0, H5=0, H6=0,
                 rutaTETIS='C:/ProgramFiles/Tetis9/'):
    """Crea el archivo Hantec de condiciones iniciales en la simulación TETIS.
    
    Entradas:
    ---------
    ruta:     string. Carpeta donde se encuentra el proyecto TETIS
    tet:      string. Nombre y extensión (.tet) del archivo del proyecto TETIS
    H1:       float. Almacenamiento estático (0-100% del máximo)
    H2:       float. Almacenamiento superficial (mm)
    H3:       float. Almacenamiento gravitacional (mm)
    H4:       float. Acuífero (mm)
    H5:       float. Caudal en el cauce (0-1000% del caudal en sección llena)
    H6:       float. Intercepción por la vegetación (0-100% del máximo)
    rutaTETIS: string. Carpeta de instalación de TETIS
    
    Salidas:
    --------
    Se genera el archivo Hantec especificado
    """
    
    # copiar archivo .tet como 'FileSSP.txt'
    os.chdir(ruta)
    copyfile(tet, 'FileSSP.txt')
    
    # extraer nombre del archivo paramgeo del archivo .tet
    with open(tet) as f:
        tetLines = f.readlines()
        tetLines = [x.strip() for x in tetLines]
    f.close()
    paramgeo = tetLines[1]
    hantec = tetLines[4]

    # leer archivo paramgeo
    with open(paramgeo) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    f.close()
    
    # modificar archivo paramgeo con los estados iniciales
    Ho = {'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4, 'H5': H5, 'H6': H6}
    for i, H in enumerate(Ho):
        lines[20 + i] = '{0:.8f}'.format(Ho[H])
        
    # sobreescribir archivo paramgeo
    with open(paramgeo, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    f.close()
        
    # copiar archivo ejecutable
    copyfile(rutaTETIS + 'bin/Hantec.exe', 'Hantec.exe')
    
    # generar estados iniciales
    print('Creando estados inciales:', hantec)
    code = os.system('Hantec.exe')
    if code != 0:
        print('ERROR. No se pudieron crear los estados iniciales.')
        return

    os.remove('Hantec.exe')
    os.remove('FileSSP.txt')

    
def crear_nieve(ruta, tet, Zo, Bnieve=1, rutaTETIS='C:/ProgramFiles/Tetis9/', verbose=True):
    """Crea el mapa de condiciones iniciales de la nieve.
    
    Entradas:
    ---------
    ruta:     string. Carpeta donde se encuentra el proyecto TETIS
    tet:      string. Nombre y extensión (.tet) del archivo del proyecto TETIS
    Zo:       int. Cota mínima a la que aparece la  nieve
    Bnieve:   float. Coeficiente para interpolar la altura de la nieve con la cota
    rutaTETIS: string. Carpeta de instalación de TETIS
    """
    
    # copiar archivo .tet como 'FileSSP.txt'
    os.chdir(ruta)
    copyfile(tet, 'FileSSP.txt')
    
    # leer archivo de proyecto
    with open(ruta + tet) as f:
        lines = [x.strip() for x in f.readlines()]
    f.close()
    nieve = lines[10]
    
    
    # MAPA DE COBERTURA DE NIEVE
    # ..........................
    
    # crear archivo defconie
    with open('defconie.txt', 'w') as f:
        f.write('{0:.0f}\n'.format(Zo))
    f.close()

    # generar estados iniciales
    copyfile(rutaTETIS + 'bin/DefCodNie.exe', 'DefCodNie.exe')
    if verbose:
        print('Creando cobertura inicial de nieve:', nieve)
    code = os.system('DefCodNie.exe')
    if code != 0:
        print('ERROR. No se pudo crear la cobertura inicial de nieve.')
        return
    
    os.remove('DefCodNie.exe')
    
    # ESPESOR DE LA NIEVE
    # ...................
    
    # extraer nombre del archivo calib a partir del archivo .tet
    with open(tet) as f:
        tetLines = [x.strip() for x in f.readlines()]
    f.close()
    calib = tetLines[2]

    # leer archivo calib
    with open(calib) as f:
        pars = [float(x.strip()) for x in f.readlines()]
    f.close()
    
    # modificar el valor del coeficiente de interpolación de la nieve
    pars[9] = Bnieve
    
    # sobreescribir archivo calib
    with open(calib, 'w') as f:
        for par in pars:
            f.write('{0:.7f}\n'.format(par))
    f.close()
    
    # interpolar espesor de la nieve
    copyfile(rutaTETIS + 'bin/IntNieves.exe', 'IntNieves.exe')
    if verbose:
        print('Interpolar espesor de la nieve:', nieve)
    code = os.system('IntNieves.exe')
    if code != 0:
        print('ERROR. No se pudo interpolar el espesor de la nieve.')
        return
    
    os.remove('FileSSP.txt')
    os.remove('IntNieves.exe')
    

def crear_calib(FCs, path, sufix=None, flow='total'):
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
    values = [1, 1, 0.2, 1, 0.4, 10, 1, 200, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 100000, 0.00065]
    idx = ['FC' + str(i) for i in range(1, 10)] + ['FC0', 'Bnieve', 'DDF1', 'DDF2', 'Tb', 'Bppt', 'USLE1', 'USLE2', 'USLE3', 'p1', 'H3max', 'Btmp']
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
    if sufix is None:
        filename = 'Calib.txt'
    else:
        filename = 'Calib_' + sufix + '.txt'
    with open(path + filename, 'w+') as f:
        for value in FCs:
            f.write(str(round(value, 7)) + '\n')
            
            
def leer_calib(calibFile):
    """Lee el archivo de parámetros de TETIS
    
    Entrada:
    --------
    calibFile: string. Archivo a leer
    
    Salida:
    -------
    calib:     pd.Series. Parámetros del modelo
    """
    
    calib = pd.read_csv(calibFile, header=None, squeeze=True)
    idx = ['FC' + str(i) for i in range(1, 10)] + ['FC0', 'Bnieve', 'DDF1', 'DDF2', 'Tb', 'Bppt', 'USLE1', 'USLE2', 'USLE3', 'p1', 'H3max', 'Btmp']
    calib.index = idx
    
    return calib


def crear_VarSCEUA(ruta, parametros, sufix=None, OF='Nash', skip=1, p_forma=2.0, n_dias=30, A_pond=0,
                   verbose=True):
    """Crea el archivo de configuración de la calibración automática del módulo hidrológico de TETIS.
    
    Entradas:
    ---------
    ruta:       string. Carpeta en la que guardar el archivo
    parametros: dict. Contiene a su vez un diccionario para cada uno de los parámetros a calibrar con el valor mínimo (min) y máximo (max) del rango de búsqueda, y la semilla (seed)
    sufix:      string. Sufijo a añadir al nombre del archivo. El nombre de archivo sería Var-SCEUA_sufix.txt
    OF:         strin. Función objetivo a emplear: RMSE, %Vol, Nash, HMLE, RMSE-monthly, ErrGA, ErrorLog, Nash-ranges, LogBiasNSE, KGE
    skip:       integer. Paso temporal de la simulación a partir del que calcular la función objetivo
    p_forma:    float. Valor del parámetro de forma para el cálculo de la función objetivo HMLE
    n_dias:     integer. Número de días en los que agrupar el caudal para el cálculo de la función objetivo RMSE mensual
    A_pond:     integer. Si ponderar (1) o no (0) la función objetivo según el área de la cuenca de los puntos aforados
    verbose:    boolean. Si se quiere mostrar el proceso por pantalla
    
    Salidas:
    --------
    Archivo 'Var-SCEUA_sufix.txt'"""
    
    # conversión del nombre de la función objetivo en su código
    OFcode = {'RMSE': 4, '%Vol': 5, 'Nash': 6, 'HMLE': 8, 'RMSE-monthly': 10, 'ErrGA': 12, 'ErrorLog': 13,
              'Nash-ranges': 20, 'LogBiasNSE': 21, 'KGE': 22}

    # TABLA POR DEFECTO DE RANGOS Y VALOR INICIAL DE LOS PARÁMETROS
    # .............................................................
    
    # definir valores mínimos, máximos y por defecto
    mins = [0.01, 0, 0, 0.001, 0, 0.001, 0, 0.001, 0, 0, 0, 0, 0, 0, 0, -0.2, 0, 0, 1, 3, -2, 0, 0]
    maxs = [3, 2, 1.5, 10, 1.5, 5000, 1, 50000, 1.5, 100, 100, 100, 100, 100, 100, 0.2, 1, 1, 3, 6, 4, 3, 1000000]
    defs = [1, 1, 0.2, 1, 0.4, 10, 0.5, 200, 1, 50, 0.5, 0.5, 0.5, 50, 0, 0, 1, 1, 1, 1, 1, 0, 100000]
    # definir el nombre de los parámetros calibrables
    # factores correctores
    idx1 = ['FC' + str(i) for i in range(1, 10)]
    # estados iniciales
    idx2 = ['H6'] + ['H' + str(i) for i in range(1, 6)]
    # otras variables
    idx3 = ['Bppt', 'Bnieve', 'FC0', 'DDF1', 'DDF2', 'Tb', 'p1', 'H3max']
    idx = idx1 + idx2 + idx3
    # generar data frame
    VarSCEUA = pd.DataFrame(data=[mins, maxs, defs], columns=idx).T
    VarSCEUA.columns = ['min', 'max', 'init']
    VarSCEUA['cal'] = 'F'
    
    # MODIFICAR LOS PARÁMETROS A CALIBRAR, SU RANGO DE BÚSQUEDA Y LA SEMILLA
    # ......................................................................
    for par in parametros:
        VarSCEUA.loc[par, :] = [parametros[par][v] for v in ['min', 'max', 'seed']] + ['T']

    # EXPORTAR ARCHIVO 'Var-SCEUA.txt'
    # ................................
    if sufix is None:
        filename = 'Var-SCEUA.txt'
    else:
        filename = 'Var-SCEUA_{0}.txt'.format(sufix)
    with open(ruta + filename, 'w+') as f:
        if verbose:
            print('Creando archivo {0}{1}'.format(ruta, filename))
        # Exportar variables de cálculo de la función objetivo
        f.write(str(skip) + '\n')
        f.write(str(OFcode[OF]) + '\n')
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


def crear_tet(ruta, sistema, fechas, met=None, flow=None):
    """Generador del archico .tet de un modelo TETIS.
    
    Entradas:
    ---------
    ruta:    string. Ruta donde se guardará el archivo .tet
    sistema:  string. Nombre (cuenca hidrográfica, estación) que aparecerá al inicio del
                     nombre del archivo .tet
    met:     string. Método de calibración secuencial
    flow:    string. Tipo de caudal: 'total', 'quick', 'slow'
    
    Salida:
    -------
    En la carpeta 'ruta' se genera el archivo 'nombre flow_met_scn.tet'"""
    
    modelname = sistema + ' ' + flow + '_' + met + '.tet'
    print(modelname)
    
    # exportar archivo .tet
    with open(ruta + modelname, 'w+') as file:
        # write attributes
        file.write(ruta +'\n')
        file.write('Paramgeo.txt\n')
        file.write('Calib_' + flow + '_' + met + '.txt\n')
        file.write('Topolco.sds\n')
        file.write('Hantec{0}.sds\n'.format(fechas[0]))
        file.write('{0}_{1}-{2}_{3}.txt\n'.format(sistema, *fechas, flow[0].upper()))
        file.write('FactorETmes_{0}.txt\n'.format(sistema.lower()))
        file.write('CurvasHV.txt\n')
        file.write('Hantec{0}.sds\n'.format(fechas[1]))
        file.write('{0}-{1}_{2}_{3}.res\n'.format(*fechas, flow, met))
        file.write('Nieve{0}.asc\n'.format(fechas[0]))
        file.write('Nieve{0}.asc\n'.format(fechas[1]))
        file.write('mdt_{0}.asc\n'.format(sistema.lower()))
        file.write('hu_{0}.asc\n'.format(sistema.lower()))
        file.write('ks_{0}.asc\n'.format(sistema.lower()))
        file.write('kp_{0}.asc\n'.format(sistema.lower()))
        file.write('cobveg_{0}.asc\n'.format(sistema.lower()))
        file.write('pte_{0}.asc\n'.format(sistema.lower()))
        file.write('fd_{0}.asc\n'.format(sistema.lower()))
        file.write('fac_{0}.asc\n'.format(sistema.lower()))
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
        file.write('ks_{0}.asc\n'.format(sistema.lower()))
        file.write('kp_{0}.asc\n'.format(sistema.lower()))
        file.write('kps_{0}.asc\n'.format(sistema.lower()))
        file.write('vel_{0}.asc\n'.format(sistema.lower()))
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


def lanzar_SCEUA(ruta, tet, rutaTETIS='C:/ProgramFiles/Tetis9/', verbose=True):
    """Lanza la calibración automática del modelo TETIS según la secuencia de calibración del método 'met'.
    
    Entradas:
    ---------
    FCs:     series. Contiene los valores de los parámetros del archivo 'Calib.txt' distintos del valor por defecto
    ruta:    string. Carpeta donde se encuentra el modelo TETIS y todos sus archivos auxiliares
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
    os.chdir(ruta)
    copyfile(tet, 'FileSSP.txt')
    
    # extraer nombres de los archivos necesarios desde el archivo .tet
    with open(ruta + tet) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    f.close()
    calibFile = lines[2]
    try:
        suffix = calibFile.split('.')[0].split('_')[0]
    except:
        suffix = None
    topolco = lines[3]
    resFile = lines[9]
    resSCEUA = lines[27]
    
    # generar el archivo TOPOLCO.sds si no existiera
    if topolco not in os.listdir():
        if verbose:
            print('Generando archivo:', topolco)
        copyfile('{0}/bin/Toparc.exe'.format(rutaTETIS), 'Toparc.exe')
        code = os.system('Toparc.exe')
        if code != 0:
            print('ERROR. No se generó el archivo TOPOLCO.')
        os.remove('Toparc.exe')
            
    # calibración automática
    if verbose:
            print('Calibrando modelo: {0}{1}'.format(ruta, tet))
    copyfile('{0}/bin/CalibAutom.exe'.format(rutaTETIS), 'CalibAutom.exe')
    code = os.system('CalibAutom.exe')
    if code != 0:
        print('ERROR. La calibración no fue correcta.')
        return
    else:
        # importar los valores calibrados de los parametros
        leer_SCEUA(ruta, resSCEUA)
        pars = leer_SCEUA.parametros
        print('NSE = ', leer_SCEUA.resultados.Nash.max())
        
        # importar los archivos 'Calib.txt' originales
        calib = pd.read_csv(ruta + calibFile, header=None)
        calib.index = ['FC' + str(i) for i in range(1, 10)] + ['FC0', 'Bnieve', 'DDF1', 'DDF2', 'Tb', 'Bppt', 'USLE1', 'USLE2', 'USLE3', 'p1', 'H3max']
        calib = calib[0]
        
        # modificar los valores de los parametros calibrados
        for FC in pars.index:
            if FC[0] == 'R':
                par_name = 'FC' + FC[1]
            elif FC[:3] == 'Cap':
                par_name = 'H3max'
            calib[par_name] = pars.loc[FC, 'optimo']

        # exportar el archivo calib modificado
        crear_calib(calib, ruta, sufix=suffix)
        lanzar_SCEUA.calib = calib
        
        # simular
        if verbose:
            print('Simulando modelo: {0}{1}'.format(ruta, tet))
        copyfile('{0}/bin/Tetis.exe'.format(rutaTETIS), 'Tetis.exe')
        code = os.system('Tetis.exe')
        if code != 0:
            print('ERROR. La simulación no fue correcta.')
            return
        else:
            leer_resultados(ruta, resFile, observed=True, plotQ=False, export=False)
            lanzar_SCEUA.obs = leer_resultados.observado
            lanzar_SCEUA.sim = leer_resultados.simulado
            nse = NSE(leer_resultados.observado, leer_resultados.simulado)
            if verbose:
                print('NSE =', round(nse, 3))
    
    os.remove('CalibAutom.exe')
    os.remove('Tetis.exe')
    os.remove('FileSSP.txt')
    

def definir_aforos(ruta, inputFile, stns=None):
    """Corrige el archivo de entrada para que sólo se tengan en cuenta las estaciones de aforo de interés.
    
    Entradas:
    ---------
    ruta:      string. Carpeta del proyecto TETIS
    inputFile: string. Archivo de entrada de TETIS
    stns:      list. Estaciones de aforo a tener en cuenta. Por defecto es 'None', en cuyo caso se utilizan todas
    """

    # leer archivo de entrada
    with open(inputFile) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    f.close()
    
    # definir estaciones de aforo a tener en cuenta
    for l, line in enumerate(lines):
        if (line[0] == 'Q'):
            if stns is None:
                continue
            stn = line.strip().split()[1][1:]
            if stn not in stns:
                # comentar la línea
                lines[l] = '*' + line
        elif (line[:2] == '*Q'):
            stn = line.strip().split()[1][1:]
            if stns is None:
                # descomentar la línea
                lines[l] = line[1:]
            elif stn in stns:
                # descomentar la línea
                lines[l] = line[1:]
            
    # reescribir el archivo de entrada
    with open(inputFile, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    f.close()
    
    
def lanzar_multievento(ruta, tet, eventos, stns=None, rutaTETIS='C:/ProgramFiles/Tetis9/', verbose=True):
    """Lanza una simulación multievento.
    
    Entradas:
    ---------
    ruta:      string. Carpeta del proyecto TETIS
    tet:       string. Nombre del archivo de proyecto (.tet)
    eventos:   list. Serie de eventos a simular. Ej: ['2010-01', '2015-01']
    stns:      list. Estaciones de aforo a tener en cuenta. Por defecto es 'None', en cuyo caso se utilizan todas
    rutaTETIS: string. Carpeta donde está instalado TETIS
    verbose:   boolean. Si se quiere mostrar el proceso por pantalla
    """
    
    # leer archivo de proyecto
    with open(ruta + tet) as f:
        lines = [x.strip() for x in f.readlines()]
    f.close()
    
    for event in eventos:
        yyyymm = ''.join(event.split('-'))
        if verbose:
            print(event)

        # cambiar archivo de entrada
        sistema = lines[5].split('_')[0]
        inputFile = '{0}_{1}_E.txt'.format(sistema, event)
        lines[5] = inputFile
        
        # modificar las estaciones de cálculo en el archivo de entrada
        definir_aforos(ruta, inputFile, stns=stns)

        # cambiar estados iniciales
        lines[4] = 'Hantec{0}.sds'.format(yyyymm)
        lines[10] = 'Nieve{0}.asc'.format(yyyymm)

        # cambiar estados finales
        lines[8] = 'Hantec{0}f.sds'.format(yyyymm)
        lines[11] = 'Nieve{0}f.asc'.format(yyyymm)

        # cambiar archivo de resultados
        lines[9] = '{0}_E.res'.format(yyyymm)

        with open(ruta + tet, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        f.close()

        # lanzar simulación
        lanzar_control(ruta, tet, rutaTETIS=rutaTETIS, verbose=False)
        lanzar_simulacion(ruta, tet, rutaTETIS=rutaTETIS, verbose=False)

def estaciones_serie_var(ruta, inputFile, var='P'):
    """Extrae del archivo de entrada las estaciones y las series para la variable indicada
    
    Entradas:
    ---------
    ruta:      string. Carpeta del proyecto TETIS
    inputFile: string. Nombre del archivo de entrada (incluida extensión .txt)
    var:       string. Código de la variable: 'P' precipitación, 'T' temperatura, 'E' evapotranspiración, 'Q' caudal, 'H' nieve
    
    Salidas:
    --------
    serie:     pd.DataFrame. Serie para cada una de las estaciones
    stns:      pd.DataFrame. Atributos de las estaciones
    nSteps:    int. Nº de pasos temporales del evento
    At:        str. Resolución temporal en minutos
    start:     datetime. Fecha de inicio del evento
    """
    
    variables = ['P', 'T', 'E', 'Q', 'H']
    
    # leer archivo de entrada
    definir_aforos(ruta, inputFile, stns=None)
    with open(ruta + inputFile) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    f.close()
    for l, line in enumerate(lines):
        if line[0] == 'G':
            nSteps, At = [int(x) for x in line.split()[1:]]
            At = '{0}min'.format(At)
        elif line[0] == 'F':
            date, hhmm = line.split()[1:]
            yyyy, MM, dd = [int(x) for x in date.split('-')][::-1]
            hh, mm = [int(x) for x in hhmm.split(':')]
            start = datetime(yyyy, MM, dd, hh, mm)
        elif line[0] in variables:
            break
            
    serie = pd.DataFrame(columns=pd.date_range(start, periods=nSteps, freq=At))
    stns = pd.DataFrame(columns=['Tipo', 'Estacion', 'X', 'Y', 'Z', 'O'])
    
    variables.remove(var)
    
    # serie y estaciones de precipitación
    for line in lines[l:]:
        if line[0] == var:
            attr = [line.split()[0], line.split()[1][1:]] + [int(x) for x in line.split()[3:7]]
            stns.loc[attr[1], :] = attr
            data = [float(x) for x in line.split()[7:]]
            serie.loc[attr[1], :] = data
    
    return serie, stns, nSteps, At, start


def generar_multievento(ruta, sistema, eventos, aforos=None, verbose=True):
    """Corrige los archivos de entrada para una calibración/simulación multievento de manera que todos tengan las mismas estaciones para cada variable.
    
    Entradas:
    ---------
    ruta:      string. Carpeta del proyecto TETIS
    sistema:  string. Nombre de la cuenca de estudio
    eventos:   list. Nombres de los eventos
    aforos:    list. Nombre de las estaciones de aforo que se quieren simular. Por defecto es 'None' y se toman todas
    verbose:   boolean
    """
    
    rutaOrig = os.getcwd()
    os.chdir(ruta)

    # encontrar archivo de entrada y condiciones iniciales para cada evento
    multievento = {}
    for event in eventos:
        multievento[event] = {}
        for p in ['C', 'E']:
            multievento[event][p] = {}
            yyyymm = ''.join(event.split('-'))
    #         multievento[event]['Hantec'] = 'Hantec{0}.sds'.format(yyyymm)
            multievento[event][p]['input'] = '{0}_{1}_{2}.txt'.format(sistema, event, p)

    # leer series y estaciones de cada variable para cada evento
    for event in eventos:
        for p in ['C', 'E']:
            inputFile = multievento[event][p]['input']
            for var in ['P', 'T', 'E', 'Q', 'H']:
                serie, stns, nSteps, At, start = estaciones_serie_var(ruta, inputFile, var=var)
                dct = {}
                dct['stns'] = stns
                dct['serie'] = serie
                multievento[event][p][var] = copy.deepcopy(dct)
                multievento[event][p]['pars'] = [nSteps, At, start]

    # conjuntos de estaciones disponibles para cada variable en algún evento
    estaciones = {}
    for var in ['P', 'T', 'E', 'Q', 'H']:
        stns = pd.DataFrame(columns=['Tipo', 'Estacion', 'X', 'Y', 'Z', 'O'])
        for event in eventos:
            for p in ['C', 'E']:
                stns = pd.concat((stns, multievento[event][p][var]['stns']), axis=0).drop_duplicates()
        stns.sort_index()
        estaciones[var] = stns
        if verbose:
            print(var, stns.shape)
        del stns

    # formato y código de dato faltante para cada variable
    formats = {'P': '%.1f ', 'T': '%.1f ', 'E': '%.2f ', 'Q': '%.2f ', 'H': '%.1f '}
    na_reps = {'P': -1, 'T': -99, 'E': -1, 'Q': -1, 'H': -1}

    # ARCHIVO DE ENTRADA DE TETIS
    for event in eventos:
        for p in ['C', 'E']:
            filename = multievento[event][p]['input']
            print('{0:<80}'.format(filename), end='\r')
            
            # exportar encabezado
            periods, timestep, start = multievento[event][p]['pars']
            timestep = int(timestep[:-3])
            export_heading(filename, start, periods, timestep)

            # exportar series
            for var in ['P', 'T', 'E', 'Q', 'H']:
                try:
                    # extraer estaciones y series de la variable
                    dct = multievento[event][p][var].copy()
                    stns, serie = dct['stns'], dct['serie']
                    # rellenar con estaciones ficticias si faltase alguna estación
                    if estaciones[var].shape[0] != serie.shape[0]:
                        missing = list(set(estaciones[var].index) - set(serie.index))
                        for stn in missing:
                            stns.loc[stn, :] = estaciones[var].loc[stn, :]
                            serie.loc[stn, :] = np.nan
                        stns.sort_index(inplace=True)
                        serie.sort_index(inplace=True)
                    # unir estaciones y serie, y corregir datos faltantes
                    frmt = formats.get(var, '%.1f ')
                    na_rep = na_reps.get(var, -1)
                    aux = pd.concat((stns, serie), axis=1, join='inner')
                    aux.replace(np.nan, na_rep, inplace=True)
                    export_series(aux, filename, format=frmt)
                except:
                    continue

            # definir estaciones de aforo de cálculo
            definir_aforos(ruta, filename, stns=aforos)
            
    os.chdir(rutaOrig)

    
def calentar_multievento(ruta, tet, sistema, eventos, Zo=3000, Bnieve=1, H1=0, H2=0, H3=0, H4=0, H5=0, H6=0,
                         rutaTETIS='C:/ProgramFiles/Tetis9/'):
    """Crea los archivos de condiciones iniciales (Hantec.sds y Nieve.asc) para cada uno de los eventos.
    
    Entradas:
    ---------
    ruta:     string. Carpeta de proyecto
    tet:      string. Nombre del archivo .tet de TETIS
    sistema:  string. Nombre de la cuenca de estudio
    eventos:  list. Nombres de los eventos
    Zo:       int. Cota mínima a la que aparece la  nieve
    Bnieve:   float. Coeficiente para interpolar la altura de la nieve con la cota
    H1:       float. Almacenamiento estático (0-100% del máximo)
    H2:       float. Almacenamiento superficial (mm)
    H3:       float. Almacenamiento gravitacional (mm)
    H4:       float. Acuífero (mm)
    H5:       float. Caudal en el cauce (0-1000% del caudal en sección llena)
    H6:       float. Intercepción por la vegetación (0-100% del máximo)
    rutaTETIS: string. Carpeta de instalación de TETIS
    """
    
    # leer archivo de proyecto
    with open(ruta + tet) as f:
        lines = ines = [x.strip() for x in f.readlines()]
    f.close()
    
    # modificar archivo de evento
    lines[4] = 'Hantec0.sds'
    lines[10] = 'Nieve0.asc'
    with open(tet, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    f.close()
        
    # crear un topolco único
#     lanzar_topolco(rutaSim, tet, rutaTETIS=rutaTETIS)

    # crear condiciones iniciales de nieve
    crear_nieve(ruta, tet, Zo=Zo, Bnieve=1, rutaTETIS=rutaTETIS)

    # crear condiciones iniciales
    crear_Hantec(ruta, tet, H1=H1, H3=H3, H4=H4, H5=H5, rutaTETIS=rutaTETIS)

    for event in eventos:
        yyyymm = ''.join(event.split('-'))

        # modificar archivo de evento
        lines[5] = '{0}_{1}_C.txt'.format(sistema, event)
        lines[8] = 'Hantec{0}.sds'.format(yyyymm)
        lines[9] = '{0}_C.res'.format(yyyymm)
        lines[11] = 'Nieve{0}.asc'.format(yyyymm)

        # sobreescribir archivo de evento
        with open(tet, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        f.close()

        # lanzar calentamiento
        lanzar_simulacion(ruta, tet, rutaTETIS=rutaTETIS)


        
def crear_multicalib(ruta, tet, sistema, eventos, rutaTETIS='C:/ProgramFiles/Tetis9/'):
    """Crea el archivo MultiCalib.txt necesario para la calibrabción multievento.
    
    Entrada:
    --------
    ruta:     string. Carpeta de proyecto
    tet:      string. Nombre del archivo .tet de TETIS
    sistema:  string. Nombre de la cuenca de estudio
    eventos:  list. Nombres de los eventos
    rutaTETIS: string. Carpeta de instalación de TETIS
    """

    # encontrar archivo de entrada y condiciones iniciales para cada evento
    multievento = {}
    for event in eventos:
        multievento[event] = {}
        yyyymm = ''.join(event.split('-'))
        multievento[event]['Hantec'] = 'Hantec{0}.sds'.format(yyyymm)
        multievento[event]['input'] = '{0}_{1}_E.txt'.format(sistema, event)

    with open(ruta + 'MultiCalib.txt', 'w') as f:
        for event in eventos:
            f.write('{0}\t{1}\n'.format(multievento[event]['Hantec'], multievento[event]['input']))
    f.close()
