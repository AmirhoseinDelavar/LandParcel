import pickle
import config
from Builder import Builder
import time

def run_location_finding(total_execution_time):
    with open('outputs/p_maps_list.pickle', 'rb') as f:
        p_maps_list = pickle.load(f)
    with open('outputs/export_map.pickle', 'rb') as f:
        export_map = pickle.load(f)

    export_map.reset_map_for_location_finding()
    
    iteration = 0
    config.log_location_finding_settings()
    config.log(f"------Location Finding------")
    start_time = time.time()
    for map in p_maps_list:
        config.log(f"----Step {iteration}----")
        config.log(f"Map:{map.map_id} Parcel:{map.parcel_id} Area:{map.curr_size} Type:{map.parcel_type}")
        building = Builder(map)
        building_mask = building.ga_run()
        if building.has_building:
            config.log(f"Parcel Has Fixed-Facilities")
        export_map.draw_building(building_mask,iteration,map.map_id,map.parcel_id,building.has_building,map.parcel_type,map.block_mask)
        iteration+=1
        config.log(f"-"*8)
    export_map.draw_building_results()
    export_map.draw_collision()
    export_map.report()

    with open('outputs/export_map.pickle', 'wb') as f:
        pickle.dump(export_map, f)

    config.log("--- Location Finding Finished In %s seconds ---" % (time.time() - start_time))
    total_execution_time += time.time() - start_time
    return total_execution_time


if __name__ == "__main__":
    run_location_finding(0)