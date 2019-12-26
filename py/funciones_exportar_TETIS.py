# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __03/07/2018__ <br>
# 
# __Introducción__<br>
# Se incluyen las funciones para generar el archivo de entrada en formato CEDEX para el modelo hidrológico TETIS.
# -  `export_heading` genera el encabezado del archivo de entrada.
# -  `export_series` exporta las series climáticas e hidrológicas en formato CEDEX.
# -  `export_control` exporta los puntos de control, que no tienen serie y por tanto no puede hacerse con `export_series`.
# 
# __Cosas a corregir__ <br>


import numpy as np

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-whitegrid')

import pandas as pd
import datetime
from simpledbf import Dbf5

import os


def export_heading(path, start_date, periods, timestep):
    """This function creates the heading of the time series file according to the requirements set by TETIS
    
    Attributes
    ----------
    path: path (inclunding file name and extension) in which the file will be exported
    start_date: datetime value that correspondes to the first record
    periods: integer. Number of timesteps in the simulation
    timestep: integer. Duration of a timestep in minutes"""
    
    with open(path, "wt") as file:
        file.write('* Número de registros (n) e intervalo temporal (At)\n')
        file.write('* {0:<5} {1:<7} \n'.format('n', 'At(min)'))
        file.write('{0:<1} {1:<5.0f} {2:<4.0f} \n'.format('G', periods, timestep))
        file.write('*\n')
        file.write('* Fecha de inicio del episodio\n')
        file.write('{0:1} {1:<10} {2:<5}\n'.format('*', 'dd-mm-aaaa', 'hh:mm'))
        file.write('{0:1} {1:<10} {2:<5}\n'.format('F', start_date.strftime('%d-%m-%Y'), start_date.strftime('%H:%M')))
        file.write('*\n')

    file.close()


def export_series(df, path, format='%.1f '):
    """This function exports a data frame (df) as a txt file according to the requirements set by TETIS
    
    Attribute
    ----------
    df: data frame with the recorded values to be exported. It must be a row file (each row represents a station) with
    the following attributes: station type, station code, x-UTM, y-UTM, elevations, and the sequence of records
    path: path (inclunding file name and extension) in which the file will be exported
    format: string. Format in which the data series will be exported. Default is floating point number with one decimal"""
    
    with open(path, "a") as file:
        # Column names
        file.write('* {0:-<12} {1:-<10} {2:-<10} {3:-<6} {4:-<4} '.format('Estación', 'X-utm', 'Y-utm', 'Z', 'O'))
        # cols = df.columns.tolist()[5:]
        # file.writelines(['%-4s' % item  for item in cols])
        file.write('\n')
        
        # Type, station, coordinates and row time series
        for i in range(df.shape[0]):
            file.write('{0: <1} "{1: <10}" {2: <10.0f} {3: <10.0f} {4: <6.0f} {5: <4.0f} '.format(df.iloc[i, 0], 
                                                                                                  df.iloc[i, 1], 
                                                                                                  df.iloc[i, 2],
                                                                                                  df.iloc[i, 3], 
                                                                                                  df.iloc[i, 4], 
                                                                                                  df.iloc[i, 5]))
            values = df.iloc[i, 6:].tolist()
            file.writelines([format % item  for item in values])
            file.write("\n")

    file.close()


def export_control(df, path):
    """This function exports the control points (included in 'df')
    
    Attributes
    ----------
    df: data frame with the recorded values to be exported. It must be a row file (each row represents a station) with
    the following attributes: station type, station code, x-UTM, y-UTM, elevations
    path: path (inclunding file name and extension) in which the file will be exported"""
    
    with open(path, "a") as file:
        # Column names
        file.write('* {0:-<12} {1:-<15} {2:-<15} {3:-<6} {4:-<4} '.format('Estación', 'X-utm', 'Y-utm', 'Z', 'O'))
        #cols = df.columns.tolist()[5:]
        #file.writelines(['%-4s' % item  for item in cols])
        file.write('\n')
        
        # Type, station, coordinates and row time series
        for i in range(df.shape[0]):
            file.write('{0: <1} "{1: <10}" {2: <15.6f} {3: <15.6f} {4: <6.0f} {5: <4.0f} '.format(df.iloc[i, 0], 
                                                                                                  df.iloc[i, 1], 
                                                                                                  df.iloc[i, 2],
                                                                                                  df.iloc[i, 3], 
                                                                                                  df.iloc[i, 4], 
                                                                                                  df.iloc[i, 5]))
            file.write("\n")

    file.close()

