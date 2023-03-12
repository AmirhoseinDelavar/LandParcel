import pickle
from Parcels import Parcels
import time
from multiprocessing import Pool
import config

def f_ga_run(map):
    return Parcels(map).ga_run()

def run_partitioning(total_execution_time):
    maps_list = []
    export_map = []
    # load maps from axis finder
    with open('outputs/maps_list.pickle', 'rb') as f:
        maps_list = pickle.load(f)
    with open('outputs/export_map.pickle', 'rb') as f:
        export_map = pickle.load(f)

    export_map.reset_map_for_partitioning()

    # partitioning on axis (1.2)
    iteration = 0
    config.log_partitioning_settings()
    config.log(f"------Partioning------")
    start_time = time.time()
    parcels_maps_list = []
    # make pool
    result_parcels = []
    config.log(f"----Parallel Processing----")
    with Pool(config.PROCESSING_CORES) as p:
        result_parcels = p.map(f_ga_run, maps_list)
    for parcel in list(result_parcels):
        config.log(f"----Step {iteration}----")
        config.log(f"Map:{parcel.map.map_id} Parcels:{parcel.map.parcel_cnt} Area:{parcel.map.curr_size}")
        config.log(f"Parcels:{parcel.report}")
        parcels_maps_list.extend(parcel.parcels_maps)
        config.log(f"Best Results:{parcel.best_solution}")
        export_map.add_partition_report(parcel.report)
        export_map.draw_partitions(iteration,parcel.map,parcel.best_points)
        iteration+=1
        config.log(f"-"*8)
    export_map.draw_partitioning_results()
    export_map.draw_collision()
    export_map.report()

    with open('outputs/p_maps_list.pickle', 'wb') as f:
        pickle.dump(parcels_maps_list, f)
    with open('outputs/export_map.pickle', 'wb') as f:
        pickle.dump(export_map, f)
    

    config.log("--- Partitioning Finished In %s seconds ---" % (time.time() - start_time))
    total_execution_time += time.time() - start_time
    return total_execution_time


if __name__ == "__main__":
    run_partitioning(0)
