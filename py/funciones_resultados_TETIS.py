
# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __12/06/2019__ <br>
# 
# __Introducción__<br>
# Se importa y analizo el archivo de resultados del modelo TETIS.
# 
# Se incluyen tres funciones para leer las salidas de TETIS y generar las series en csv y los mapas en ASCII:
# -  `leer_caudal` genera las series de caudal observado y simulado en formato 'data frame' a partir del archivo de resultados de caudal.
# -  `leer_sedimento` genera las series de caudal sólido observado y simulado en formato 'data frame' a partir del archivo de resultados de sedimentos.
# -  `agregar_ascii` lee los mapas generados para una variable y los agrega en un único mapa.
# -  `corregir_ascii` corrige los mapas cuando tienen valores tan grandes que no entran en el espacio reservado en el archivo de resultados y son guardados como "************".
# 
# __Cosas a corregir__ <br>
# 
# 
# __Índice__<br>
# 
# [1. Lectura de resultados](#1.-Lectura-de-resultados)<br>
# [2. Criterios de rendimiento](#2.-Criterios-de-rendimiento)<br>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import matplotlib.gridspec as gridspec
import datetime
from calendar import monthrange
import sys

import os
sys.path.append('F:/Codigo/Python/Calibrar/')
from funciones_rendimiento import NSE, sesgo

from scipy.interpolate import griddata


# ### 1. Lectura de resultados


def leer_caudal(path, file, observed=True, plot=True, export=True, verbose=True):
    """Lee el archivo de resultados de la simulación y genera las series de caudal simulado (observado) para las estaciones de aforo y los puntos de control.
    
    Parámetros:
    -----------
    path:      string. Ruta donde se encuentra el archivo de resultados y donde se exportará la serie generada.
    file:      string. Nombre del archivo de resultados con extensión (.res).
    observed:  boolean. Si hay observaciones de caudal. En ese caso se extraen las series observadas.
    plot:      boolean. Si se quieren plotear las series generadas.
	export:    boolean. Si se quieren exportar las series generadas.
    verbose:   boolean. Si se quiere mostrar en pantalla la dimensión de las series calculadas.
    
    Salidas:
    --------
    leer_caudal:   object. Consistente en una serie de métodos que contienen la información extraída:
                           observado: dataframe. Serie de caudal observado (m³/s)
                           simulado:  dataframe. Serie de caudal simulado (m³/s)
                           embalses:  list. Listado del código de los embalses, si los hubiera.
                           aforos:    list. Listado del código de las estaciones de aforo, si las hubiera.
                           control:    list. Listado del código de los puntos de control, si los hubiera.
    Los dos 'data frame' se pueden exportar en formato csv en la carpeta 'path' con los nombres 'Qobs_añoinicio-añofin.csv' y 'Qsim_añoinicio-añofin.csv'.
    Además, se puede generar un gráfico con las series de caudal observado y simulado."""
        
    # Abrir la conexión con el archivo de resultados y leer todas las líneas en la lista 'res'
    with open(path + file) as r:
        res = r.readlines()
        res = [x.strip() for x in res]
    r.close()
    
    # Extraer la fecha del inicio de la simulación como "datetime.date"
    start = pd.to_datetime(res[14].split()[1], dayfirst=True).date()
    # Extraer el número de pasos temporales
    timesteps = int(res[17].split()[1])
    
    # Extraer las estaciones de aforo, las de control y la línea en la que empieza a aparecer la serie
    embalses = []
    aforos = []
    control = []
    for i in range(len(res)):
        line = res[i].split()
        if (line[0] == 'N'):
            embalses.append(line[1][1:])
        if line[0] == 'Q':
            aforos.append(line[1][1:])
        if line[0] == 'B':
            control.append(line[1][1:])
        if len(line) > 1:
            if line[1] == '------DT---':
                break
    leer_caudal.embalses, leer_caudal.aforos, leer_caudal.control = embalses, aforos, control
    skiprows = i
    del i
    
    # Generar la serie de caudal
    # --------------------------
    # Importar la serie
    usecols = 2 + 7*len(embalses) + 2*len(aforos) + len(control)
    if verbose == True:
        print('Nº embalses:', len(embalses), '\tNº aforos:', len(aforos), '\tNº control:', len(control))
    results = pd.read_csv(path + file, delim_whitespace=True,
                         skiprows=skiprows, nrows=timesteps, header=0, 
                         usecols=range(usecols), index_col=0)
    # Corregir columnas
    results.index.name = results.columns[0]
    cols = list(results.columns[1:])
    results = results.iloc[:,:-1]
    results.columns = cols
    # Extraer las series de caudal de la serie de resultados
    icols = []
    for i, col in enumerate(results.columns):
        if col[8] in ['I', 'Q', 'B']:
            icols.append(i)
    caudal = results.iloc[:, icols]
    caudal.replace(to_replace=-1, value=np.nan, inplace=True)
    # Definir índices
    caudal.index = pd.date_range(start=start, periods=timesteps)
    caudal.index.name = 'fecha'

    # Dividir la serie en observado y simulado
    if observed == True:
        obscols, simcols = [], []
        for i in range(len(embalses) + len(aforos)):
            obscols.append(i * 2)
            simcols.append(1 + i * 2)
        for i2 in range(len(control)):
            simcols.append(i2 + (i + 1)*2)
        # Extraer serie de caudal observado
        caudal_obs = caudal.iloc[:, obscols]
        caudal_obs.columns = embalses + aforos
        if verbose == True:
            print('Dimensión serie observada:\t', caudal_obs.shape)
        # Extraer serie de caudal simulado
        caudal_sim = caudal.iloc[:, simcols]
        caudal_sim.columns = embalses + aforos + control
        if verbose == True:
            print('Dimensión serie simulada:\t', caudal_sim.shape)
    else:
        caudal_sim = caudal
        caudal_sim.columns = control
        if verbose == True:
            print('Dimensión serie simulada:\t', caudal_sim.shape)

    # Plotear las series
    if plot == True:
        plt.figure(figsize=(18, 5))
        if observed == True:
            plt.plot(caudal_obs, linewidth=1, c='steelblue', alpha=0.3)
        plt.plot(caudal_sim, linewidth=1, c='maroon', alpha=0.15)
        plt.xlim((caudal_sim.index[0], caudal_sim.index[-1]))
        if observed == True:
            ymax = np.ceil(max(caudal_sim.max().max(), caudal_obs.max().max()) / 100) * 100
        else:
            ymax = np.ceil(caudal_sim.max().max() / 100) * 100
        plt.ylim((0, ymax))
        plt.ylabel('Q (m³/s)', fontsize=12);
    
    # Exportar las series
    if not os.path.exists(path + 'resultados/series/caudal/'):
        os.makedirs(path + 'resultados/series/caudal/')
    output = '_' + str(caudal_sim.index[0].year) + '-' + str(caudal_sim.index[-1].year) + '.csv'
    if observed == True:
        caudal_obs.to_csv(path + 'resultados/series/caudal/Qobs' + output, float_format='%.1f')
    caudal_sim.to_csv(path + 'resultados/series/caudal/Qsim' + output, float_format='%.1f')
    
    if observed == True:
        leer_caudal.observado = caudal_obs
    leer_caudal.simulado = caudal_sim


def leer_embalses(path, file, observed=True, plot=True):
    """Lee el archivo de resultados de la simulación y genera las series de caudal simulado (observado) para las estaciones
    de aforo y los puntos de control.
    
    Parámetros:
    -----------
    path:      string. Ruta donde se encuentra el archivo de resultados y donde se exportará la serie generada.
    file:      string. Nombre del archivo de resultados con extensión (.res).
    observed:  boolean. Si hay observaciones de caudal sólido. En ese caso se extraen las series observadas.
    plot:      boolean. Si se quieren plotear las series generadas.
    
    Salidas:
    --------
    caudal_obs:   dataframe. Serie de caudales observados (m³/s). Sólo si 'observed' es True.
    caudal_sim:   dataframe. Serie de caudales simulados (m³/s).
    Estos dos 'data frame' se exportan en formato csv en la carpeta 'path' con los nombres 'Qobs_añoinicio-añofin.csv'
    y 'Qsim_añoinicio-añofin.csv'."""
        
    # Abrir la conexión con el archivo de resultados y leer todas las líneas en la lista 'res'
    with open(path + file) as r:
        res = r.readlines()
        res = [x.strip() for x in res]
    r.close()
    
    # Extraer la fecha del inicio de la simulación como "datetime.date"
    start = pd.to_datetime(res[14].split()[1], dayfirst=True).date()
    # Extraer el número de pasos temporales
    timesteps = int(res[17].split()[1])
    
    # Extraer las estaciones de aforo, las de control y la línea en la que empieza a aparecer la serie
    embalses = []
    for i in range(len(res)):
        line = res[i].split()
        if line[0] == 'N':
            embalses.append(line[1][1:])
        if len(line) > 1:
            if line[1] == '------DT---':
                break
    skiprows = i
    del i
    
    # Generar la serie de caudal
    # --------------------------
    # Importar la serie
    usecols = 2 + 7*len(embalses)
    results = pd.read_csv(path + file, delim_whitespace=True,
                         skiprows=skiprows, nrows=timesteps, header=0, 
                         usecols=range(1, usecols), index_col=0)

    # Generar series de los embalses (si así se especifica)
    icols = []
    for i, col in enumerate(results.columns):
        if col[8] in ['N', 'I', 'V', 'S']:
            icols.append(i)
    reserv = results.iloc[:, icols]
    # Definir el encabezado de las columnas
    cols = [] # nombre de las columnas
    for stn in embalses:
        cols += [stn + '_No', stn + '_Ns', stn + '_Ib', stn + '_Is', stn + '_Vo', stn + '_Vs', stn + '_So']
    # Definir índices
    reserv.columns = cols
    reserv.index = pd.date_range(start=start, periods=timesteps)
    reserv.index.name = 'fecha'

    # Dividir la serie en observado y simulado
    obscols, simcols = [], []
    for i in range(len(embalses)):
        obscols += [i*5, 2 + i*5, 4 + i*5]
        simcols += [1 + i*5, 3 + i*5]
    reserv_obs = reserv.iloc[:, obscols]
    reserv_sim = reserv.iloc[:, simcols]
    
    # Plotear las series
    if plot == True:
        plt.figure(figsize=(18, 5))
        if observed == True:
            plt.plot(caudal_obs, linewidth=1, c='steelblue', alpha=0.3, legend='observado')
        plt.plot(caudal_sim, linewidth=1, c='maroon', alpha=0.15, legend='simulado')
        plt.xlim((caudal_sim.index[0], caudal_sim.index[-1]))
        ymax = np.ceil(max(caudal_sim.max().max(), caudal_obs.max().max()) / 100) * 100
        plt.ylim((0, ymax))
        plt.ylabel('Q (m³/s)', fontsize=12)
        plt.legend();
        
    # Exportar las series
    output = '_' + str(caudal_sim.index[0].year) + '-' + str(caudal_sim.index[-1].year) + '.csv'
    if observed == True:
        caudal_obs.to_csv(path + 'resultados/series/caudal/Qobs' + output, float_format='%.1f')
    caudal_sim.to_csv(path + 'resultados/series/caudal/Qsim' + output, float_format='%.1f')
    
    if observed == True:
        if reservoir == True:
            return caudal_obs, caudal_sim, reserv_obs, reserv_sim
        else:
            return caudal_obs, caudal_sim
    else:
        if reservoir == True:
            return caudal_sim, reserv_sim
        else:
            return caudal_sim


def leer_sedimento(path, file, observed=True, plot=True):
    """Lee el archivo de resultados de la simulación de sedimentos y genera las series simuladas para los puntos
    de control de sedimentos, los aforos de caudal y los puntos de control.
    
    Parámetros:
    -----------
    path:      string. Ruta donde se encuentra el archivo de resultados y donde se exportará la serie generada.
    file:      string. Nombre del archivo de resultados con extensión (.txt).
    observed:  boolean. Si hay observaciones de caudal sólido. En ese caso se extraen las series observadas.
    plot:      boolean. Si se quieren plotear las series generadas.
    
    Salidas:
    --------
    sed_obs:   dataframe. Serie de caudales de sedimento observados (m³/s). Sólo si 'observed' es True.
    sed_sim:   dataframe. Serie de caudales de sedimento simulados (m³/s).
    Estos dos 'data frame' se exportan en formato csv en la carpeta 'path' con los nombres 'Sobs_añoinicio-añofin.csv'
    y 'Ssim_añoinicio-añofin.csv'."""
    
    # Abrir la conexión con el archivo de resultados y leer todas las líneas en la lista 'res'
    with open(path + file) as r:
        res = r.readlines()
        res = [x.strip() for x in res]
    r.close()
    
    # Extraer la fecha del inicio de la simulación como "datetime.date"
    start = pd.to_datetime(res[14].split()[1], dayfirst=True).date()
    # Extraer el número de pasos temporales
    timesteps = int(res[17].split()[1])
    # Generar las fechas
    dates = pd.date_range(start, periods=timesteps)
    
    # Extraer las estaciones de aforo de sedimento, aforo líquido y control (si éstas existieran)
    afo_q = []
    control = []
    afo_s = []
    for i in range(len(res)):
        line = res[i].split()
        if line[0] == 'Q':
            try:
                line_Q
            except:
                line_Q = i
            afo_q.append(line[1][1:])
        if line[0] == 'B':
            try:
                line_B
            except:
                line_B = i
            control.append(line[1][1:])
        elif line[0] == 'X':
            try:
                line_X
            except:
                line_X = i
            afo_s.append(line[1][1:])
        if len(line) > 1:
            if (line[1] == '----DT----') or (line[1] == '------DT---'):
                break
    leer_sedimento.afo_q, leer_sedimento.control, leer_sedimento.afo_s = afo_q, control, afo_s
    # Línea del archivo en la que empieza la primera serie
    skiprows = i + 2
    
    # Crear el(los) 'data frame' vacío donde se guardarán las series simuladas(observadas)
    sed_sim = pd.DataFrame(data=np.nan, index=dates, columns=afo_s + afo_q + control)
    sed_sim.index.name = 'fecha'
    if observed == True:
        sed_obs = pd.DataFrame(data=np.nan, index=dates, columns=afo_s)
        sed_obs.index.name = 'fecha'
    
    # Extraer las series en las estaciones de aforo de sedimentos
    # -----------------------------------------------------------
    for j in range(len(afo_s)):
        for i, l in enumerate(range(skiprows, skiprows + timesteps)):
            if observed == True:
                sed_obs.iloc[i, j] = float(res[l].split()[1])
            sed_sim.iloc[i, j] = float(res[l].split()[2])
    
    # Extraer las series en las estaciones de aforo de caudal
    # -------------------------------------------------------
    for j in range(len(afo_q)):
        # Encontrar la fila en la que empieza la serie de la estación 'j'
        for i in range(len(res)):
            line = res[i].split()
            if len(line) > 1:
                if line[1] == '------DT---':
                    break
        skiprows = i + 1
        # Extraer la serie de la estación 'j'
        for i, l in enumerate(range(skiprows, skiprows + timesteps)):
            sed_sim.iloc[i, j + len(afo_s)] = float(res[l].split()[5])
    
    # Extraer las series en los puntos de control
    # -------------------------------------------
    # Encontrar la fila en la que empiezan las series
    for i in range(len(res)):
        line = res[i].split()
        if len(line) > 1:
            if line[1] == '------DT--':
                break
    skiprows = i + 1
    # Extraer las series
    for i, l in enumerate(range(skiprows, skiprows + timesteps)):
        for j in range(len(control)):
            k = j + len(afo_s) + len(afo_q)
            sed_sim.iloc[i, k] = float(res[l].split()[j + 1])
    
    # Plotear las series
    if observed == True:
        print('Dimensión serie observada:\t', sed_obs.shape)
    print('Dimensión serie simulada:\t', sed_sim.shape)
    if plot == True:
        plt.figure(figsize=(18, 5))
        if observed == True:
            plt.plot(sed_obs, linewidth=1, c='steelblue', alpha=0.5, label='observado')
            ymax = np.ceil(max(sed_sim.max().max(), sed_obs.max().max()) / 100) * 100
        else:
            ymax = np.ceil(sed_sim.max().max() / 100) * 100
        plt.plot(sed_sim, linewidth=1, c='maroon', alpha=0.15, label='simulado')
        plt.xlim((sed_sim.index[0], sed_sim.index[-1]))
        plt.ylim((0, ymax))
        plt.ylabel('Qsed (m³/s)', fontsize=12)
        #plt.legend(fontsize=12);
        
    # Exportar las series
    if not os.path.exists(path + 'resultados/series/sedimento/'): #comprobar si existe la carpeta de salida
        os.makedirs(path + 'resultados/series/sedimento/')
    output = '_' + str(sed_sim.index[0].year) + '-' + str(sed_sim.index[-1].year) + '.csv' # archivo de salida
    if observed == True:
        sed_obs.to_csv(path + 'resultados/series/sedimento/Sobs' + output, float_format='%.3f')
    sed_sim.to_csv(path + 'resultados/series/sedimento/Ssim' + output, float_format='%.3f')
    
    if observed == True:
        leer_sedimento.observado = sed_obs
        leer_sedimento.simulado = sed_sim
    else:
        leer_sedimento.simulado = sed_sim


def agregar_ascii(path, variable='H0', period=None, aggregation='mean', format='%12.5f '):
    """Lee todos los mapas de la variable 'variable' dentro de la carpeta 'path', los agrega según la función especificada
    y los exporta a un nuevo archivo ASCII en la misma carpeta con el formato indicado.
    
    Parámetros:
    -----------
    path:        string. Ruta de la carpeta del proyecto TETIS.
    variable:    string. Variable que se quiere estudiar. Debe coincidir con el inicio del nombre de los archivos ASCII.
    period:      list of integers. Años inicial y final del periodo de estudio
    aggregation: string. Función de agregación de los mapas: 'mean' o 'sum'.
    format:      string. Formato en el que se quieren exportar los datos.
    
    Salidas:
    --------
    Se exporta un ASCII dentro de la carpeta con el nombre 'variable_añoinicio-añofin_aggregation.asc'.
    agg_map:     array. Un 'array' 2D con el mapa agregado."""
    
    # Leer los archivos de la carpeta respectivos a la variable
    n = len(variable)
    var_files = []
    # seleccionar archivos de la variable indicada
    for file in os.listdir(path + '_ASCII/'): 
        if file[:n] == variable:
            var_files.append(file)
    # seleccionar archivos dentro del periodo de estudio
    if period != None:
        var_files2 = []
        for i, file in enumerate(var_files):
            year = int(file[n + 1: n + 5])
            if (year > period[0]) & (year <= period[1]):
                var_files2.append(file)
        start = str(period[0])
        end = str(period[1])
        var_files = var_files2
        del var_files2
    else:
        start = var_files[0][n + 1: n + 5]
        end = var_files[-1][n + 1: n + 5]
            
    # Abrir la conexión con un archivo de resultados y leer todas las líneas en la lista 'res'
    with open(path + '_ASCII/' + var_files[0]) as r:
        res = r.readlines()
        res = [x.strip() for x in res]
    r.close()

    # Extraer las coordenadas, dimensión, etc...
    ncols = int(res[0].split()[1])
    nrows = int(res[1].split()[1])
    xllcorner = float(res[2].split()[1])
    yllcorner = float(res[3].split()[1])
    cellsize = float(res[4].split()[1])
    NODATA_value = int(res[5].split()[1])
    del res
    
    # 3D 'array' con los mapas de la variable
    maps = np.empty([len(var_files), nrows, ncols])
    for i, file in enumerate(var_files):
        maps[i::] = np.loadtxt(path + '_ASCII/' + file, skiprows=6)
    
    # Crear el mapa agregado según la función especificada
    if aggregation == 'mean':
        agg_map = maps.mean(axis=0)
    elif aggregation == 'sum':
        agg_map = maps.sum(axis=0)
    # Corregir el mapa según la variable
    if variable == 'X1': # convertir precipitación media diaria en precipitación acumulada anual
        agg_map[agg_map >= 0] = agg_map[agg_map >= 0] * 365
    elif variable == 'P4': # convertir la erosión de m³/celda a m y a valores positivos
        agg_map[agg_map != NODATA_value] = -1 * agg_map[agg_map != NODATA_value] / cellsize**2
    
    # Exportar el mapa como ASCII
    if not os.path.exists(path + 'resultados/mapas/'):
        os.makedirs(path + 'resultados/mapas/')
    output = path + 'resultados/mapas/' + variable + '_' + start + '-' + end + '_' + aggregation + '.asc'
    with open(output, "wt") as file:
        # Escribir el encabezado
        file.write('{0:<14} {1:<6d}\n'.format('ncols', ncols))
        file.write('{0:<14} {1:<6d}\n'.format('nrows', nrows))
        file.write('{0:<14} {1:<12.5f}\n'.format('xllcorner', xllcorner))
        file.write('{0:<14} {1:<12.5f}\n'.format('yllcorner', yllcorner))
        file.write('{0:<14} {1:<12.5f}\n'.format('cellsize', cellsize))
        file.write('{0:<14} {1:<6d}\n'.format('NODATA_value', NODATA_value))
        # Escribir la serie
        for row in range(agg_map.shape[0]):
                file.writelines([format % item  for item in agg_map[row, :]])
                file.write("\n")
    file.close()


def corregir_ascii(path, variable='H0', format='%12.5f '):
    """Lee todos los mapas de la variable 'variable' dentro de la carpeta 'path', encuentra si hay algún dato erróneo en el
    que la magnitud es superior al espacio de la columna y por tanto se guarda como '************' y lo completa como la
    media de las celdas alrededor.
    
    Parámetros:
    -----------
    path:        string. Ruta de la carpeta del proyecto TETIS.
    variable:    string. Variable que se quiere estudiar. Debe coincidir con el inicio del nombre de los archivos ASCII.
    format:      string. Formato en el que se quieren exportar los datos.
    
    Salidas:
    --------
    Se sobreescribe el ASCII corregido."""
    
    # Leer los archivos de la carpeta respectivos a la variable
    var_files = []
    for file in os.listdir(path + '_ASCII/'):
        if file[:len(variable)] == variable:
            var_files.append(file)
    
    for file in var_files:
        # Abrir la conexión con un archivo de resultados y leer todas las líneas en la lista 'res'
        with open(path + '_ASCII/' + file) as r:
            res = r.readlines()
            res = [x.strip() for x in res]
            res = [x.split() for x in res]
        r.close()

        # Extraer las coordenadas, dimensión, etc...
        ncols = int(res[0][1])
        nrows = int(res[1][1])
        xllcorner = float(res[2][1])
        yllcorner = float(res[3][1])
        cellsize = float(res[4][1])
        NODATA_value = int(res[5][1])

        # Extraer el mapa, convertir el error en NaN y todos los datos en números reales
        data = np.array(res[6:])
        data[data == '************'] = np.nan
        data = data.astype(float)
        del res

        # Encontrar las posiciones con NaN y modificar esos datos con la media de los valores a su alrededor
        pos = np.where(np.isnan(data) == True)
        for error in range(len(pos[0])):
            i = pos[0][error]
            j = pos[1][error]
            data[i, j] = np.nanmean(data[i-1:i+2, j-1:j+2])

        # Exportar el mapa como ASCII
        with open(path + '_ASCII/' + file, "wt") as f:
            # Escribir el encabezado
            f.write('{0:<14} {1:<6d}\n'.format('ncols', ncols))
            f.write('{0:<14} {1:<6d}\n'.format('nrows', nrows))
            f.write('{0:<14} {1:<12.5f}\n'.format('xllcorner', xllcorner))
            f.write('{0:<14} {1:<12.5f}\n'.format('yllcorner', yllcorner))
            f.write('{0:<14} {1:<12.5f}\n'.format('cellsize', cellsize))
            f.write('{0:<14} {1:<6d}\n'.format('NODATA_value', NODATA_value))
            # Escribir la serie
            for row in range(data.shape[0]):
                    f.writelines([format % item  for item in data[row, :]])
                    f.write("\n")
        f.close()


def leer_SCEUA(path, file, plot=True, s=6):
    """Lee el archivo de resultados de la calibración y crea dos 'data frames' con los resultados.
    
    Parámetros:
    -----------
    path:      string. Ruta donde se encuentra el archivo de resultados y donde se exportará la serie generada.
    file:      string. Nombre del archivo de resultados con extensión (.txt).
    plot:      boolean. Si se quieren plotear un diagrama de dispersión con el valor de la función objetivo en función
               del valor normalizado de cada uno de los parámetros calibrados
    s:         integer. Tamaño de los puntos del diagrama de dispersión
    
    Salidas:
    --------
    leer_SCEUA:   object. Consistente en una serie de métodos que contienen la información extraída:
                           parametros: dataframe. Contiene los parámetros calibrados, el rango de variación, el valor
                                       inicial y el valor optimizado
                           SCEUA:      dataframe. Contiene los datos de cada una de las iteraciones de la calibración
    Además, se puede generar un diagrama de dispersión del valor de la función objetivo para cada parámetro 
    (normalizado de 0 a 1)"""

    # Abrir la conexión con el archivo de resultados y leer todas las líneas en la lista 'res'
    with open(path + file) as r:
        res = r.readlines()
        res = [x.strip() for x in res]
    r.close()

    # Encontrar número de parámetros
    for l, line in enumerate(res):
        if line == 'Obj. Funct.':
            # número de parámetros
            npar = l - 5
            break

    # Tabla con los parámetros calibrados, su rango de variación y el valor inicial
    rangos = pd.DataFrame(data=None, columns=['min', 'max', 'inicial'])
    for i in range(5, l):
        aux = res[i].split()
        par = aux[0]
        rang = [float(x) for x in aux[1:]]
        rangos.loc[par,:] = rang
    rangos.index = [i.replace('R', 'FC') if i[0] == 'R' else 'H3max' if i[:3] == 'Cap' else i for i in rangos.index]
    
    # Nombre de las funciones objetivo
    OFs = []
    for i in range(l + 1, len(res)):
        if res[i] == '':
            break
        OFs.append(res[i].split()[0])

    # La función objetivo seleccionada es:
    OF = res[l + 17].split()[-1]
    if OF == 'Nash':
        OF = 'NSE'
    elif OF == '%Vol':
        OF = 'sesgo'
    else:
        OF = 'sesgo'

    # Ha sido calculada para:
    period = [int(res[l + 18].split()[-3]), int(res[l + 18].split()[-1])]

    # Número de cuencas simuladas
    aux = [float(x) for x in res[l + 20].split()] # primera línea de los resultados
    ncuen = int((len(aux) - npar - 2) / len(OFs))

    # Crear 'data frame' con los resultados del SCEUA
    cols = list(rangos.index)
    cols.append(OF)
    for f in OFs:
        if ncuen > 1:
            for c in range(ncuen):
                cols.append(f + '_' + str(c + 1))
        else:
            cols.append(f)
    SCEUA = pd.DataFrame(columns=cols)
    SCEUA.index.name = 'iter'

    # Leer y guardar los resultados línea a línea
    for i, l2 in enumerate(range(l+20, len(res))):
        SCEUA.loc[i+1,:] = [float(x) for x in res[l2].split()[:-1]]

    # encontrar el conjunto de parámetros óptimo
    if OF == 'NSE':
        bestIdx = SCEUA[OF].sort_values(ascending=True).index[-1]
    elif OF == 'sesgo':
        bestIdx = SCEUA[OF].abs().sort_values(ascending=True).index[0]
    rangos['optimo'] = SCEUA.loc[bestIdx, rangos.index]

    # Normalizar los parámetros
    parnorm = []
    for p in rangos.index:
        pnorm = p + 'norm'
        SCEUA[pnorm] = (SCEUA[p] - rangos.loc[p,'min']) / (rangos.loc[p,'max'] - rangos.loc[p,'min'])
        parnorm.append(pnorm)

    # Visualizar:
    if plot == True:
        ncols, nrows = min(npar, 2), int(np.ceil(npar/2))
        # Configurar la figura
        plt.figure(figsize=(5 * ncols, 5 * nrows))
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)#, left=0.05, right=0.48, wspace=0.05)

        for i, pnorm in enumerate(parnorm):
            row = int(np.floor(i/ ncols))
            col = i % ncols
            ax = plt.subplot(gs[row, col])
            ax.scatter(SCEUA[pnorm], SCEUA[OF], s=s)#, c='steelblue')
            bestx, bestOF = SCEUA.loc[bestIdx, pnorm], SCEUA.loc[bestIdx, OF] 
            ax.scatter(bestx, bestOF, s=4*s)
            if OF == 'NSE':
                ylim = (-1, 1)
            elif OF == 'sesgo':
                ylim = (-100, 100)
            ax.set(xlim=(0, 1), ylim=ylim)
            ax.set_xlabel(pnorm, fontsize=12)
            ax.set_ylabel(OF, fontsize=12)

    leer_SCEUA.parametros = rangos
    leer_SCEUA.resultados = SCEUA
    leer_SCEUA.mejorOF = SCEUA.loc[bestIdx, OF]



def rendimiento(observed, simulated, monthly=True, aggregation='mean', skip=None, steps=None):
    """Sobre un par de series diarias (observada y simulada) genera la serie mensual según el método de agregración 
    definido. Posteriormente, sobre estas dos series, calcula tres criterios de rendimiento: el coeficiente de eficiencia de Nash-
    Sutcliffe (NSE), el sesgo en volumen (%V) y la raíz del error cuadrático medio (RMSE) para cada cuenca en las series
    y los coloca en un 'data frame' que posteriormente exporta.
    
    Parámetros:
    -----------
    observed:    dataframe. Serie observada; cada fila representa una fila y cada columna una estación.
    simulated:   dataframe. Serie simulada; cada fila representa una fila y cada columna una estación.
    monthly:     boolean. Si se quieren calcular los estadísticos sobre la serie mensual (por defecto) o no.
    aggregation: string. Si 'monthly' es True, tipo de agregación de los datos diarios a mensuales.
    skip:        integer. Número de pasos temporales a evitar en el cálculo de los criterios por considerarse 
                 calentamiento. Si 'monthly' es True, 'skip' se refiere al número de meses.
    steps:       integer. Número de meses a tener en cuenta a partir de 'skipmonths'. Si 'monthly' es True, 'skip' se 
                 refiere al número de meses.
    
    Salidas:
    --------
    Se genera un objeto llamado 'rendimiento' con los siguientes métodos:
    obs_month:       dataframe. Serie agregada mensual de las observaciones. Sin recortar
    sim_month:       dataframe. Serie agregada mensual de la simulación. Sin recortar
    performance: dataframe. Valores de los tres criterios para cada estación y el periodo de estudio"""
    
    # Recortar las series para que tengan la misma longitud
    if observed.shape[0] != simulated.shape[0]:
        min_ind = max(observed.index[0], simulated.index[0])
        max_ind = min(observed.index[-1], simulated.index[-1])
        observed = observed.loc[min_ind:max_ind, :]
        simulated = simulated.loc[min_ind:max_ind, :]
    
    # Calcular las series mensuales
    if monthly == True:
        obs = observed.groupby(by=[observed.index.year, observed.index.month]).aggregate([aggregation])
        sim = simulated.groupby(by=[simulated.index.year, simulated.index.month]).aggregate([aggregation])
        # Corregir los índices
        dates = []
        for date in obs.index:
            dates.append(datetime.datetime(year=date[0], month=date[1], day=monthrange(date[0], date[1])[1]))
        obs.index = dates
        sim.index = dates
        del dates
        # Corregir los nombres de las columnas
        obs.columns = obs.columns.levels[0]
        sim.columns = sim.columns.levels[0]
        # Guardar las series mensuales
        rendimiento.obs_month = obs
        rendimiento.sim_month = sim
    else:
        obs = observed
        sim = simulated
    
    # 'data frame' donde se guardará el rendimiento de cada cuenca
    performance = pd.DataFrame(index=['NSE', '%V', 'RMSE'], columns=observed.columns)
    performance.index.name = 'criterio'

    # Calcular el rendimiento de cada cuenca
    for stn in observed.columns:
        # 'data frame' con la observación y la simulación de una estación
        data = pd.concat((obs[stn], sim[stn]), axis=1)
        data.columns = ['obs', 'sim']
        #return data
        # Recortar la serie al periodo de estudio
        if skip is not None:
            if steps is None:
                data = data.iloc[skip:, :]
            else:
                data = data.iloc[skip:(skip + steps), :]
        # Calcular criterios
        performance.loc['NSE', stn] = NSE(data)
        performance.loc['%V', stn] = sesgo(data)
        performance.loc['RMSE', stn] = RMSE(data)
        del data
    
    #return obs, sim, performance
    rendimiento.performance = performance


# __Gráficas__


def hidrograma(simulado, stn, observado=None, xlim=None, 
               labels=['observado', 'simulado']):
    """Genera un gráfico con el hidrograma para las dos series de entrada y la estación indicada.
    
    Parámetros:
    -----------
    simulado:  data frame. Contiene las serie temporal de caudales simulados (o serie 2). Uno de sus columnas será 
               'stn'
    stn:       string/integer. Nombre de la columna de 'observado' y 'simulado' que se mostrará
    observado: data frame. Contiene la serie temporal de caudales obesrvados (o serie 1). Uno de sus columnas será 
               'stn'. Por defecto 'None', es decir, no hay serie obesrvada
    xlim:      list of dates. Dos fechas con el inicio y final del hidrograma
    labels     list of strings. Los nombres de ambas series
    
    Salidas:
    --------
    Un gráfico con dos líneas correspondientes a los dos hidrogramas."""

    plt.figure(figsize=(18, 5))
    if observado is not None:
        plt.plot(observado.loc[:, stn], c='steelblue', linewidth=1, alpha=1, label=labels[0])
        ymax = np.ceil(max(observado.loc[:, stn].max(), simulado.loc[:, stn].max()) / 10) * 10
    else:
        ymax = np.ceil(simulado.loc[:, stn].max() / 10) * 10
    plt.plot(simulado.loc[:, stn], c='maroon', linewidth=0.8, alpha=1, label=labels[1])
    if xlim is not None:    
        plt.xlim(xlim)
    plt.ylabel('Q (m³/s)', fontsize=13)
    plt.ylim((0, ymax))
    plt.title(stn, fontsize=14)
    plt.legend(fontsize=13);


def excedencia(data, col):
    """Calcular la probabilidad de excedencia de una serie temporal.
    
    Parámetros:
    -----------
    data:      data frame. Contiene las series temporales y al menos una columna llamada 'col'.
    col:       string/integer. Nombre de la columna sobre la que calcular la excedencia.
    
    Salidas:
    --------
    exceedance: data frame. Una data frame de dos columnas, una con una copia de data[col], y otra llamada 'exc' con
                la probabilidad de excedencia del valor de la serie en cada paso temporal.
    """
    
    exceedance = pd.DataFrame(data[col]).copy()
    exceedance.sort_values(col, axis=0, ascending=False, inplace=True)
    exceedance['exc'] = np.linspace(100/exceedance.shape[0], 100, num=exceedance.shape[0])
    
    return exceedance
