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
    #estaciones = pd.DataFrame(columns=['Tipo', 'cod', 'X', 'Y', 'Z'])
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
            if (line[1] == '----DT----') or (line[1] == '------DT--'):
                break
    # Línea del archivo en la que empieza la primera serie
    skiprows = i + 2
    l = i
    # Guardar las estaciones
    leer_sedimento.afo_q, leer_sedimento.control, leer_sedimento.afo_s = afo_q, control, afo_s
    print('Nº aforos caudal:', len(afo_q), '\tNº control:', len(control), '\tNº aforos sedimento:', len(afo_s))

    
    # Crear el 'data frame' con las estaciones y sus coordenadas
    # ----------------------------------------------------------
    # Seleccionar la línea donde empieza
    #if line_Q in locals():
        #line = line_Q
    #elif line_B in locals():
        #line = line_B
    #elif line_X in locals():
        #line = line_X
    # Extraer los datos
    #sed_qb = pd.read_csv(path + file, delim_whitespace=True, skiprows=line, nrows=len(afo_q) + len(control), header=None)
    #sed_qb.columns = ['Tipo', 'cod', 'X', 'Y', 'Z']
    #sed_qb.set_index('cod', drop=False, inplace=True)
    
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
        for i in range(l, len(res)):
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
    for i in range(l, len(res)):
        line = res[i].split()
        if len(line) > 1:
            if line[1] == '------DT--':
                break
    skiprows = i + 1
    # Extraer las series
    for i, l in enumerate(range(skiprows, skiprows + timesteps)):
        #print(i, l)
        #print()
        for j in range(len(control)):
            #print(j)
            k = j + len(afo_s) + len(afo_q)
            #print(k)
            sed_sim.iloc[i, k] = float(res[l].split()[j + 1])
    
    # Plotear las series
    if observed == True:
        print('Dimensión serie observada:\t', sed_obs.shape)
    print('Dimensión serie simulada:\t', sed_sim.shape)
    if plot == True:
        plt.figure(figsize=(18, 5))
        if observed == True:
            plt.plot(sed_obs, linewidth=1, c='steelblue', alpha=0.5, label='observado')
            ymax = ceil(max(sed_sim.max().max(), sed_obs.max().max()) / 100) * 100
        else:
            ymax = ceil(sed_sim.max().max() / 1) * 1
        plt.plot(sed_sim, linewidth=1, c='maroon', alpha=0.15, label='simulado')
        plt.xlim((sed_sim.index[0], sed_sim.index[-1]))
        plt.ylim((0, ymax))
        plt.ylabel('Qsed (m³/s)', fontsize=12);
        
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