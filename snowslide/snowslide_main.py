import os
from datetime import datetime
from snowslide.functions import *
from snowslide.display_2Dresults import *

# Module logger
import logging

log = logging.getLogger(__name__)

# from snowslide.display_3Dresults import * Don't need it for now


def snowslide_complete(
    path_dem,
    snd0,
    epsilon=0.001,
    param_simul={
        "simul_name": "glacier",
        "save_path": None,
        "save_fig": False,
        "save_array": True,
        "plot": "2D",
    },
    param_expo={"a": 0.14, "c": 145, "min": 0.05},
    param_routing={"routing": "mfd", "preprocessing": True},
    param_camera={"azimuth": 0, "elevation": 45, "distance": "auto", "facteur": 10},
    fix_colorbar={"limit": False, "vmin": 0, "vmax": None},
):
    """This function operates the gravitationnal transport of the snow from an initial map of snow heights and a dem
    Snowslide_complete is the snowslide algorithm with the most complete display functionalities allowed for the user.

    Parameters
    ----------
    path_dem: str
        Path to the dem file (.tif)
    snd0: numpy.ndarray
        Numpy matrix of the same size as the dem containing the initial snow depths (derived from precipitation).
    epsilon: float
        Condition to get out the loop (convergence is considered reached when indicateur < epsilon). Default is 0.001.
    resolution: float
        Resolution of the dem assuming the pixels are square. Default is 30. (in meters)
    param_simul:dictionnary
        Outputs recording parameters chosen by the user
        simul_name: str
            Name of the simulation. Useful for folder names if several simulations are launched. Default is 'glacier'.
        save_path: str
            Path where the data will be stored.
        save_fig: Bolean
            True if the user wants to record the snow heights at each iteration in the form of an png image.
        save_array: Bolean
            True if the user wants to store the SND matrices at each iteration in a matrix.
        plot: str
            '3D' or '2D' are the two option to display snow heights matrices.

    param_expo: dictionary
        The maximum snow height each pixel can store is computed as an exponential function of the slope. Parameters are the following:
            SND_max=c*exp(a*slope)
        a: float
            Default is -0.14.
        c: float
            Default is 145
        min: float
            Default is 0.05.
    param_routing: dictionary
        Parameters used by pysheds to realize the routing of the snow.
        routing: str
            routing method ('mfd' or 'd8')
        preprocessing: bolean
            activate or deactivate preprocessing of the DEM. Deactivate it affects convergence.
    param_camera: dictionary
        Settings that allow the user to control the 3D display offered by mayavi
        param_camera['factor']: float, multiplication of snow heights to make them easier to see. Default is 10.
        param_camera['cellsize']: float, resolution of the DEM in meters. Default is 30.
        param_camera['azimuth']: 0:360, horizontal viewing angle in degrees. Default is 0.
        param_camera['elevation']: -90:90, vertical viewing angle in degrees. Default is 45.
        param_camera['distance']: float, distance of the camera to the object. Default is 'auto'.
    More information about viewing parameters can be find in the following mayavi documentation:
    'https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html'

    fix_colorbar: dictionary
        This fix the range of the colorbar to make the animation easier to read (with a fixed caption)
        fix_colorbar['limit']: bolean
            Activate limit_colorbar or not
        fix_colorbar['vmin']: float
            minimum value displayed
        fix_colorbar['vmax']: float
            maximum value displayed

    Returns
    -------
    SND:
        final matrix of the snow height (np matrix in meters)
    SND_tot:
        list of matrices of snow height at each iteration (list of np matrices)
    convergence:
        value of indicateur at each iteration (list)
    """

    ### initialization ###
    iter = 0
    indicateur = float("inf")

    # useful dem information for registering output as a .tif file
    get_info = rasterio.open(path_dem)
    resolution = float(get_info.res[0])
    crs = str(get_info.crs)
    transform = get_info.transform

    # Importation of dem using pysheds
    grid = Grid.from_raster(path_dem)
    dem = grid.read_raster(path_dem)

    # initialization of snow quantities
    snd_tot = []
    snd = np.copy(snd0)
    snd_tot.append(np.copy(snd))

    print("Variables have been initialized")

    # Creation of the folder in which the simulation will be saved
    if param_simul["save_path"] == None:
        save_path = os.getcwd()
    else:
        save_path = param_simul["save_path"]
    simul_name = f'Snowslide_{datetime.date.today()}_{param_simul["simul_name"]}_simul{param_simul["plot"]}_routing_{param_routing["routing"]}'
    simul_folder = save_path + "/" + simul_name
    if not os.path.exists(simul_folder):
        os.makedirs(simul_folder)
    if param_simul["plot"] == "2D":
        plots_path = simul_folder + "/2D_plots"
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
    if param_simul["plot"] == "3D":
        plots_path = simul_folder + "/3D_plots"
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

    print("The simulation folder has been created, launching simulation...")

    ### Core part of the code ###

    convergence = []
    iter = 0
    while indicateur > epsilon:
        snd1 = snd.copy()
        hnso = dem + snd
        flow_dir = dem_flow(
            hnso, grid, param_routing["routing"], param_routing["preprocessing"]
        )
        hnso_slope = slope(hnso, resolution, resolution)
        snd_max = snd_max_exponential(
            hnso_slope, param_expo["a"], param_expo["c"], param_expo["min"]
        )
        snd = snow_routing(snd, snd_max, flow_dir, param_routing["routing"])

        ### Exit conditions ###

        # 1st exit condition
        indicateur = np.sum(
            (snd - snd1) ** 2
        )  # Exit condition when the L2 norm of the distance
        # between the matrices of two iterations converge towards 0.
        convergence.append(indicateur)
        snd_tot.append(np.copy(snd))

        # 2nd exit condition if indicateur converge towards a constant that is not 0.
        if iter > 5:
            speed = np.sum(np.gradient(np.array(convergence)[-5:]))
            if speed < 0 and abs(speed) < 1e-2:
                indicateur = 0

                ### Saving parameters ###

        if param_simul["save_fig"] == True:
            if param_simul["plot"] == "2D":
                legend = "Snow Depth (m)"
                title = f"SND à {iter} itération"
                xlabel = "longitude"
                ylabel = "latitude"
                figure_name = f'/figure{param_simul["plot"]}_SND_iter{iter}.png'
                save_path = plots_path + figure_name
                save_plots2D(
                    snd_tot[iter],
                    save_path=save_path,
                    legend=legend,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    fix_colorbar=fix_colorbar,
                )

            if param_simul["plot"] == "3d":
                figure_name = f'/figure{param_simul["plot"]}_SND_iter{iter}.png'
                save_path = plots_path + figure_name
                save_plots3D(
                    path_dem,
                    SND_plot=snd_tot[iter],
                    save_path=save_path,
                    param_camera=param_camera,
                )

        # Makes iterations evolve
        iter = iter + 1

    print("The algorithm converged in :", iter, " iterations")

    # Create a gif animation out of the stored images
    if param_simul["save_fig"] == True:
        gif_name = f"Snowslide_{param_simul['simul_name']}_animation.gif"
        fig_names = f"figure{param_simul['plot']}_SND_iter"
        n = np.shape(snd_tot)[0] - 1
        create_giff(
            plots_path,
            dst_path=simul_folder,
            gif_name=gif_name,
            fig_names=fig_names,
            n=n - 1,
        )

    # Save the matrices iterations in a numpy file
    if param_simul["save_array"] == True:
        data = np.copy(snd_tot)
        array_name = "/SND_at_all_iterations"
        np.save(simul_folder + array_name, data)

    # Saving the output SND as .tif file with same projection and properties as the dem
    output_path = simul_folder + "/converged_SND.tif"
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=snd.shape[0],
        width=snd.shape[1],
        count=1,
        dtype=snd.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(snd, 1)

    print(f"The files have been stored in the following folder {simul_folder}")
    print("The outputs of snowslide_complete function are : SND, SND_tot, convergence.")

    return snd, snd_tot, convergence


def snowslide_base(
    path_dem,
    snd0,
    save_path=None,
    epsilon=1e-3,
    param_expo={"a": 0.14, "c": 145, "min": 0.05},
    param_routing={"routing": "mfd", "preprocessing": True, "compute_edges": True},
    glacier_id="",
    propagation_boolean=True
):
    """This function operates the gravitationnal transport of the snow from an initial map of snow heights and a dem.
    Snowslide_base is the fastest computing snowslide algorithm with very basic display functionalities allowed for the user.

    Parameters
    ----------
    path_dem: str
        Path to the dem file (.tif)
    snd0: numpy.ndarray
        Numpy matrix of the same size as the dem containing the initial snow depths (derived from precipitation).
    save_path: str
        Path where the user want to save the data produced (.tif file of the snd matrix). Default is None.
        If no path is indicated, no data is stored (only registered as a variable when running the function)
        If a path is indicated the matrix will be saved as 'converged_SND.tif' file
    epsilon: float
        Condition to get out the loop (convergence is considered reached when indicateur < epsilon). Default is 1e-3.
    param_expo: dictionary
        The maximum snow height each pixel can store is computed as an exponential function of the slope. Parameters are the following:
            snd_max=c*exp(a*slope)
        a: float
            Default is -0.14.
        c: float
            Default is 145
        min: float
            Default is 0.05.
    param_routing: dictionary
        Parameters used by pysheds to realize the routing of the snow.
        routing: str
            routing method ('mfd' or 'd8')
        preprocessing: bolean
            activate or deactivate preprocessing of the DEM. Deactivate it affects convergence.
        compute_edges: bolean
            activate or deactivate computing of edges for the slope function
    glacier_id : str, optional
        for logging only: add the name of the glacier to the log messages
    propagation_boolean : boolean
        for reduction of snow holding depth for pixels receiving avalanches (as in Queno et al., 2023: https://doi.org/10.5194/egusphere-2023-2071)

    Returns
    -------
    Final matrix of the snow height (np matrix in meters). It is stored in the folder if indicated by the user
    """

    ### Initialization ###
    iter = 0
    indicateur = float("inf")

    # useful dem information for registering output as a .tif file
    get_info = rasterio.open(path_dem)
    crs = str(get_info.crs)
    resolution = float(get_info.res[0])
    transform = get_info.transform

    # Importation of dem using pysheds
    grid = Grid.from_raster(path_dem)
    dem = grid.read_raster(path_dem)

    # initialization of initial snow depths
    snd = np.copy(snd0)

    log.debug(
        f"{glacier_id}variables have been initialized, launching the simulation..."
    )

    ### Core part of the code ###

    convergence = []
    while indicateur > epsilon:
        snd1 = np.copy(snd)
        hnso = (
            dem + snd
        )  # total elevation surface (HNSO) is recalculated at each iteration
        flow_dir = dem_flow(
            hnso, grid, param_routing["routing"], param_routing["preprocessing"]
        )
        hnso_slope = slope(hnso, resolution, resolution, param_routing["compute_edges"])
        snd_max = snd_max_exponential(
            hnso_slope, param_expo["a"], param_expo["c"], param_expo["min"]
        )
        # reduce snow depth threshold by 30% for pixels that have revived snow, i.e. pixels tht have more snow than snd_max after the first iteration
        snd_flat = snd.flatten()
        snd_max_flat = snd_max.flatten()
        if (iter > 0) & propagation_boolean:
            snd_max_flat[snd_flat>snd_max_flat] = 0.70*snd_max_flat[snd_flat>snd_max_flat]
        snd_max_updated = snd_max_flat.reshape(snd_max.shape)
        snd_max = snd_max_updated

        snd = snow_routing(snd, snd_max, flow_dir, param_routing["routing"])

        iter = iter + 1

        ### Exit conditions ###

        # 1st exit condition
        indicateur = np.sum(
            (snd - snd1) ** 2
        )  # Exit condition when the L2 norm of the distance
        # between the matrices of two iterations converge towards 0.
        convergence.append(indicateur)

        # 2nd exit condition if indicateur converge towards a constant that is not 0.
        if iter > 5:
            speed = np.sum(np.gradient(np.array(convergence)[-5:]))
            if speed < 0 and abs(speed) < epsilon:
                indicateur = 0

    log.info(f"{glacier_id}the algorithm converged in {iter} iterations")

    # Saving the output as .tif file with same projection and properties as the dem
    if save_path != None:
        output_path = save_path + "/converged_SND.tif"
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=snd.shape[0],
            width=snd.shape[1],
            count=1,
            dtype=snd.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(snd, 1)

        log.debug(
            f"{glacier_id}the file has been stored in the following location: {output_path}"
        )

    return snd

