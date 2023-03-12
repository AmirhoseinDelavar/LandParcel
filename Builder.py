from Map import MapIn
import config
import cv2
import numpy as np
import pygad
import math

class Builder:
    ff_mask = []
    points = []
    building = []
    map = []
    ppcenter = []
    ppcenter_p = []
    building_dist = 0
    def __init__(self,map:MapIn):
        Builder.ff_mask = cv2.imread('outputs/facility_mask.bmp',cv2.IMREAD_GRAYSCALE)
        Builder.roads_mask = cv2.imread('outputs/roads_mask.bmp',cv2.IMREAD_GRAYSCALE)
        Builder.map = map
        self.has_building = False
        self.ga_instance = Builder.ga_config()
        # make points array
        # get first line center
        y1, x1 = map.line_p[0]
        y2, x2 = map.line_p[1]
        y3, x3 = map.parcel_center
        px, py = (x2-x1,y2-y1)
        dAB = px*px + py*py
        u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
        # (y,x)
        Builder.ppcenter = (int(y1 + u * py),int(x1 + u * px))
        # get parent line center
        y1, x1 = map.parent_line_p[0]
        y2, x2 = map.parent_line_p[1]
        y3, x3 = map.parcel_center
        px, py = (x2-x1,y2-y1)
        dAB = px*px + py*py
        u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
        # (y,x)
        Builder.ppcenter_p = (int(y1 + u * py),int(x1 + u * px))
        self.ppline = self.map.line_mask_maker(Builder.ppcenter,self.map.parcel_center)
        self.ppline_p = self.map.line_mask_maker(Builder.ppcenter_p,self.map.parcel_center)
        self.ppline = self.ppline & self.map.block_mask
        self.ppline_p = self.ppline_p & self.map.block_mask
        # make points arr
        # convert lines to loc
        self.lines = self.ppline | self.ppline_p
        line_arr = np.asarray(np.where(self.lines==255))
        Builder.points = list(zip(line_arr[0],line_arr[1]))
        Builder.points.sort(key= lambda x: math.dist(x,map.parcel_center))

        # select and load building
        building = config.building_sel(map.parcel_type)
        building = cv2.imread(building)
        building = cv2.cvtColor(building,cv2.COLOR_BGR2GRAY)
        # Builder.building_dist = len(building[building<255])/3
        Builder.building = building

    def ga_config():
        return pygad.GA(num_generations=20,
                       num_parents_mating=4,
                       fitness_func=Builder.fitness_func,
                       sol_per_pop=10,
                       num_genes=1,
                       init_range_low=0.0,
                       init_range_high=0.8,
                       random_mutation_max_val=0.5,
                       random_mutation_min_val=-0.5,
                       gene_space=list(np.arange(0.0, 0.8, 0.02)),
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type=None,
                       mutation_type="random",
                       allow_duplicate_genes=False,
                       mutation_percent_genes=100)

    
    def fitness_func(solution,sol_indx):
        # build
        mask_one,b_point = Builder.build(solution)
        # distance fitness
        f_axis = math.dist(b_point,Builder.ppcenter)
        s_axis = math.dist(b_point,Builder.ppcenter_p)
        dist_fitness = min(f_axis,s_axis)
        # check house border collision
        if dist_fitness < config.BUILDING_BORDER_GAP + Builder.building_dist:
            dist_fitness = -1 * config.TYPE_A_AREA
        else:
            dist_fitness = -1 * dist_fitness
        # check if building is out of border
        out_border = np.bitwise_not(Builder.map.block_mask.astype(np.uint8)) & np.bitwise_not(mask_one.astype(np.uint8))
        out_border = -4*config.TYPE_A_AREA if np.sum(out_border[out_border>0])>0 else 1

        # check if constructed on road
        on_road = Builder.roads_mask.astype(np.uint8) & np.bitwise_not(mask_one.astype(np.uint8))
        on_road = -4*config.TYPE_A_AREA if np.sum(on_road)>0 else 1
        
        # cut trees
        tress_cut = Builder.map.trees_mask.astype(np.uint8) & np.bitwise_not(mask_one.astype(np.uint8))
        carbon_fitness = -1 * (np.sum(tress_cut)/255)

        weights = config.L_CARBON_WEIGHT + config.L_DIST_WEIGHT + config.L_OUT_BORDER_WEIGHT + config.L_ON_ROAD_WEIGHT
        score = config.L_DIST_WEIGHT*dist_fitness + config.L_OUT_BORDER_WEIGHT*out_border + config.L_CARBON_WEIGHT*carbon_fitness +\
            config.L_ON_ROAD_WEIGHT * on_road
        return score/weights

    def build(solution):
        # select point
        indx = int(solution*len(Builder.points))
        b_point = Builder.points[indx]
        y_offset,x_offset = b_point
        # check nearest line
        nearset_line = []
        f_axis = math.dist(b_point,Builder.ppcenter)
        s_axis = math.dist(b_point,Builder.ppcenter_p)
        if f_axis > s_axis:
            nearset_line = Builder.map.parent_line_p
        else:
            nearset_line = Builder.map.line_p
        # placing
        s_img = Builder.building
        l_img = Builder.map.block_mask
        (h, w) = s_img.shape
        (cX, cY) = (w // 2, h // 2)
        # place image to center
        x_offset,y_offset = (x_offset-cX,y_offset-cY)
        # angle
        p0 = nearset_line[0]
        p1 = nearset_line[1]
        deltaX , deltaY = (p0[1]-p1[1],p0[0]-p1[0])
        angle = math.atan2(deltaY, deltaX)
        angle = angle*180/math.pi
        # angle = int(angle/90)*90
        # if Builder.map.dir == 1:
            # angle = -angle
        # rotate around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(M[0,0]) 
        abs_sin = abs(M[0,1])

        # find the new width and height bounds
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        M[0, 2] += bound_w/2 - cX
        M[1, 2] += bound_h/2 - cY

        # rotate image with the new bounds and translated rotation matrix
        s_img = cv2.warpAffine(s_img, M, (bound_w, bound_h),borderValue=255)
        s_img = cv2.threshold(s_img, 125, 255, cv2.THRESH_BINARY)[1]
        # build
        mask_one = np.ones(l_img.shape)
        mask_one[::] = 255
        mask_one[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

        return mask_one,b_point

    def ga_run(self):
        map = self.map
        # check if has building
        mask = map.block_mask & Builder.ff_mask
        if len(mask[mask>0]) > 0:
            self.has_building = True
            mask_one = np.ones(map.block_mask.shape)
            mask_one[::] = 255
            self.result = mask_one
            return self.result
        # run genetic algorithm
        self.ga_instance.run()
        config.log(f"Result:{self.ga_instance.best_solution()}")
        self.result,_ = Builder.build(self.ga_instance.best_solution()[0],)
        return self.result

