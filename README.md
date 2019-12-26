# TETIS
Funciones en Python para trabajar con el modelo hidrológico TETIS.

Hay dos carpetas con código:
* [__py__](https://github.com/casadoj/TETIS/tree/master/py) contiene los archivos _.py_. Éstos son los archivos con las funciones estables, es decir, aquellas que hasta el momento funcionan.
    * [calibracion_automatica.py](https://github.com/casadoj/TETIS/blob/master/py/calibracion_automatica.py) es el programa que permite hacer la calibración automática secuencial del modelo TETIS.
    * [calibracion_automatica2.py](https://github.com/casadoj/TETIS/blob/master/py/calibracion_automatica2.py) es la mejora del programa de calibración de TETIS que permite escoger el método secuencial de calibración.
    * [funciones_calibracion_TETIS](https://github.com/casadoj/TETIS/blob/master/py/funciones_calibracion_TETIS.py) contiene las funciones utilizadas por los dos programas anteriores para crear los archivos necesarios en la calibración (`crear_calib`, `crear_VarSCEUA`, `crear_tet`) y lanzar la calibración (`lanzar_SCEUA`).
    * [funciones_exportar_TETIS](https://github.com/casadoj/TETIS/blob/master/py/funciones_exportar_TETIS.py) contiene las funciones necesarias para crear el archivo de entrada de TETIS: `export_heading`, `export_series` y `export_control`.
    * [funciones_resultados_TETIS](https://github.com/casadoj/TETIS/blob/master/py/funciones_resultados_TETIS.py) contiene funciones para leer los archivos de resultados generados por TETIS, tanto del archivo .res como de los ascii con los mapas y analizar el rendimiento: `leer_caudal`, `leer_embalses`, `leer_sedimento`, `agregar_ascii` y `corregir_ascii`. También incluye una función para leer el archivo con los resultados de la calibración automática (`leer_SCEUA`), otra funciones para analizar el rendimiento numéricamente (`rendimiento`) y gráficamente (`hidrograma`, `excedencia`).
* [__ipynb__]() contiene los _notebook_ con versiones de alguno de los archivos _.py_ utilizados para adaptar la función a casos específicos.
