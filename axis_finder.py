from Map import MapIn,MapOut
from Axis import AxisFinder
import cv2
import pickle
import config
import time

def run_axis_finder(total_execution_time):
    maps_list = []
    lines_list = []
    axis_division_cond = True
    maps_processing_queue = []
    map_id = 0
    iteration = 0
    # axis finding process (1.1)
    config.log_axis_finding_settings()
    config.log(f"------Axis Finder------")
    start_time = time.time()
    while axis_division_cond:
        config.log(f"----Step {iteration}----")
        if (map_id == 0):
            # read map and make masks
            mymap = MapIn(config.MAIN_MAP_ADDR,config.MAIN_MAP_FILLED_BLOCK_MASK,config.MAIN_MAP_FILLED_F_F_MASK,
                            config.PARCELS_COUNT,config.ARCH,map_id=map_id)
            # set random values for carbon
            # mymap.correct_input()
        else:
            mymap = maps_processing_queue.pop(0)
        
        # makes axis_finder object to split
        myaxis = AxisFinder(mymap)
        # Arch Selection on start division is the same
        if mymap.arch_choice == config.ArchStyles.Customized:
            # axis_center = myaxis.get_center(mymap.block_mask)
            # mymap.set_map_axis_center(axis_center)
            cv2.imshow(f'Map{mymap.map_id}', mymap.frame)
            cv2.setMouseCallback(f'Map{mymap.map_id}', AxisFinder.click_callback)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            p_f = AxisFinder.clicked_points.pop(0)
            p_s = AxisFinder.clicked_points.pop(0)
            points=[]
            points.append((1,p_f,p_s,p_f,p_s))
        elif map_id == 0 and mymap.arch_choice == config.ArchStyles.Human_Centered_AI:
            p_f = ()
            if config.ARCH_FIRST_INPUT == None:
                cv2.imshow(f'Map{mymap.map_id}', mymap.frame)
                cv2.setMouseCallback(f'Map{mymap.map_id}', AxisFinder.click_callback)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                p_f = AxisFinder.clicked_points.pop(0)
            else:
                p_f = config.ARCH_FIRST_INPUT
            points = myaxis.iterate_throughall(mymap,p_f)
        elif map_id == 0:
            points = myaxis.iterate_throughall(mymap)
        else:
            points = myaxis.iterate_old_boundries_new_york()
        
        # dump axis if hit fixed_facility
        if points[-1][0] == -1:
            if not maps_processing_queue:
                axis_division_cond = False
            continue

        # draw the found line carbon_fitness
        config.log(f"Map div best score:{points[-1][0]} map_id:{mymap.map_id}")
        config.log(f"Map div best access split fitness:{points[-1][5][0]}")
        config.log(f"Map div best area split fitness:{points[-1][5][1]}")
        config.log(f"Map div best fixed facilities fitness:{points[-1][5][2]}")
        config.log(f"Map div best carbon fitness:{points[-1][5][3]}")

        lines_list.append((points[-1][3:5],mymap.map_id))
        # make map for split two parts (add line as access)
        split_mask_u,split_mask_d = mymap.line_split_mask_maker(points[-1][1],points[-1][2])
        line_mask = mymap.line_mask_maker(points[-1][1],points[-1][2])
        up_map = MapIn(split_mask_u,mymap,line_mask,map_id+1,0,points[-1][1:3])
        down_map = MapIn(split_mask_d,mymap,line_mask,map_id+2,1,points[-1][1:3])
        parcels = AxisFinder.cal_split_parcels(up_map,down_map,mymap.parcel_cnt)
        up_map.parcel_cnt,down_map.parcel_cnt = parcels
        config.log(f"Map:{map_id+1} Parcels:{up_map.parcel_cnt}")
        config.log(f"Map:{map_id+2} Parcels:{down_map.parcel_cnt}")
        # check finishing condition
        up_feasibility, down_feasibility = (up_map.isfeasible(), down_map.isfeasible())
        # add new maps to processing queue
        if up_feasibility:
            config.log(f"Map:{map_id+1} Added!")
            up_map.set_line_point(points[-1][1:])
            maps_processing_queue.append(up_map)
        # was the last axis so add axis to final answer
        else:
            config.log(f"Map:{map_id+1} Added As Answer")
            up_map.set_line_point(points[-1][1:])
            maps_list.append(up_map)
        if down_feasibility:
            config.log(f"Map:{map_id+2} Added!")
            down_map.set_line_point(points[-1][1:])
            maps_processing_queue.append(down_map)   
        else:
            config.log(f"Map:{map_id+2} Added As Answer")
            down_map.set_line_point(points[-1][1:])
            maps_list.append(down_map)  
            
        if not maps_processing_queue:
            axis_division_cond = False
        map_id += 2
        iteration+=1
        config.log('-'*8)

    # sort maps by size
    maps_list.sort(key=lambda x: x.curr_size,reverse=True)
    cv2.destroyAllWindows()
    # save maps
    with open('outputs/maps_list.pickle', 'wb') as f:
        pickle.dump(maps_list, f)
    with open('outputs/lines_list.pickle', 'wb') as f:
        pickle.dump(lines_list, f)

    # export split result
    config.log(f"------Axis Result------")
    export_map = MapOut(config.MAIN_MAP_ADDR,lines_list)
    export_map.draw_axis()
    export_map.draw_collision()
    export_map.report()
    # save
    with open('outputs/export_map.pickle', 'wb') as f:
        pickle.dump(export_map, f)

    config.log("--- Axis Finder Finished In %s seconds ---" % (time.time() - start_time))
    total_execution_time += time.time() - start_time
    return total_execution_time

def gradio_inference(access_ratio,start_point,carbon_weight):
    config.ACCESS_RATIO = float(access_ratio)
    config.ARCH_FIRST_INPUT = tuple(start_point)
    config.A_CARBON_WEIGHT = int(carbon_weight)
    run_axis_finder(0)
    image = cv2.imread(f'outputs/collision_map.bmp')
    return image

if __name__ == "__main__":
    run_axis_finder(0)