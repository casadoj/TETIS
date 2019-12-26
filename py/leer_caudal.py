def leer_caudal(path, file, observed=True, plot=True):
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
    leer_caudal:   object. Consistente en una serie de métodos que contienen la información extraída:
                           observado: dataframe. Serie de caudal observado (m³/s)
                           simulado:  dataframe. Serie de caudal simulado (m³/s)
                           embalses:  list. Listado del código de los embalses, si los hubiera.
                           aforos:    list. Listado del código de las estaciones de aforo, si las hubiera.
                           contro:    list. Listado del código de los puntos de control, si los hubiera.
                    Los dos 'data frame' se exportan en formato csv en la carpeta 'path' con los nombres 'Qobs_añoinicio-añofin.csv'
                    y 'Qsim_añoinicio-añofin.csv'.
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
    print('Nº embalses:', len(embalses), '\tNº aforos:', len(aforos), '\tNº control:', len(control))
    results = pd.read_csv(path + file, delim_whitespace=True,
                         skiprows=skiprows, nrows=timesteps, header=0, 
                         usecols=range(usecols), index_col=0)
    # Corregir columnas
    results.index.name = results.columns[0]
    cols = list(results.columns[1:]) 
    cols.append(results.columns[-1][:-2] + str(int(results.columns[-1][-2:]) + 1))
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
    obscols, simcols = [], []
    for i in range(len(embalses) + len(aforos)):
        obscols.append(i * 2)
        simcols.append(1 + i * 2)
    i = len(obscols) + len(simcols)
    for i2 in range(len(control)):
        simcols.append(i2 + i)
    if observed == True:
        caudal_obs = caudal.iloc[:, obscols]
        caudal_obs.columns = embalses + aforos
        print('Dimensión serie observada:\t', caudal_obs.shape)
    caudal_sim = caudal.iloc[:, simcols]
    caudal_sim.columns = embalses + aforos + control
    print('Dimensión serie simulada:\t', caudal_sim.shape)

    # Plotear las series
    if plot == True:
        plt.figure(figsize=(18, 5))
        if observed == True:
            plt.plot(caudal_obs, linewidth=1, c='steelblue', alpha=0.3)
            ymax = ceil(max(caudal_sim.max().max(), caudal_obs.max().max()) / 100) * 100
        else:
            ymax = ceil(caudal_sim.max().max() / 100) * 100
        plt.plot(caudal_sim, linewidth=1, c='maroon', alpha=0.15)
        plt.xlim((caudal_sim.index[0], caudal_sim.index[-1]))    
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
