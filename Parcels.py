from cmath import sqrt
import cv2
import numpy as np
from Map import MapIn
import config
import pygad
import math
from sympy import Point, Line, Circle
import copy

class Parcels:
    boundaries_arr = []
    last_axis_points = []
    b_indx_range = []
    img_shape = []
    map = []
    ax_indx_range = []
    axis_line_arr = []
    circle_line_len = 0
    parcels_setup = []
    def __init__(self,map:MapIn) -> None:
        Parcels.map = map
        self.parcels = map.parcel_cnt
        Parcels.last_axis_points = map.line_points
        Parcels.img_shape = map.boundry_mask.shape
        boundary = self.map.boundry_mask
        boundary = np.asarray(np.where(boundary==255))
        boundary = list(zip(boundary[0], boundary[1]))
        Parcels.boundaries_arr = boundary
        # make range tuples (boundries)
        indx = list(range(0,len(boundary),int(len(boundary)/(self.parcels+1))))
        if len(indx) - self.parcels == 2:
            indx.pop(0)
            indx.pop()
        elif len(indx) - self.parcels == 1:
            indx.pop(0)
        Parcels.b_indx_range = list(zip(indx[::2],indx[1::2]))
        Parcels.b_indx_range.extend(list(zip(indx[1::2],indx[2::2])))
        Parcels.b_indx_range.sort(key=lambda x: x[0])
        # draw line for last axis
        img = map.new_access_line & map.block_mask
        axis_line_arr = np.asarray(np.where(img==255))
        axis_line_arr = list(zip(axis_line_arr[0],axis_line_arr[1]))
        # draw line for parent axis
        img = map.parent_access_line & map.block_mask
        axis_line_arr_p = np.asarray(np.where(img==255))
        axis_line_arr_p = list(zip(axis_line_arr_p[0],axis_line_arr_p[1]))
        # select the largest line to draw pp lines
        if (len(axis_line_arr)>len(axis_line_arr_p)):
            Parcels.axis_line_arr = axis_line_arr
        else:
            Parcels.axis_line_arr = axis_line_arr_p
        # make range tuples (axis)
        indx = list(range(0,len(Parcels.axis_line_arr),int(len(Parcels.axis_line_arr)/(self.parcels+1))))
        if len(indx) - self.parcels == 2:
            indx.pop(0)
            indx.pop()
        elif len(indx) - self.parcels == 1:
            indx.pop(0)
        Parcels.ax_indx_range = list(zip(indx[::2],indx[1::2]))
        Parcels.ax_indx_range.extend(list(zip(indx[1::2],indx[2::2])))
        Parcels.ax_indx_range.sort(key=lambda x: x[0])
        # circle line len
        Parcels.circle_line_len = sqrt((map.frame_shape[0]**2)+(map.frame_shape[1]**2))
        # genetic algorithm defining
        self.ga_instance = 0

    def line_split_maker(img,p0:tuple,p1:tuple):
        # points are (y,x) oriented
        img_pixels = img.shape[0]*img.shape[1]
        img_x = img.shape[1]
        # numpy image is (y*x*3)
        # !! unit16 may not be enough
        y_index = (np.arange(img_pixels).reshape(img.shape[:2])/img_x).astype(np.uint32)
        x_index = np.arange(img_pixels).reshape(img.shape[:2])%img_x
        if p1[1] == p0[1]:
            up_down_line = x_index - p0[1]
        else:
            slope = (p1[0]-p0[0])/(p1[1]-p0[1]) 
            intercept = p0[0] - (slope*p0[1])
            up_down_line = x_index*slope + intercept - y_index
        # down part (becareful about center)
        down_mask = np.where(up_down_line>=0,255,0).reshape(img.shape[:2])
        up_mask = np.where(up_down_line>=0,0,255).reshape(img.shape[:2])
        return (up_mask,down_mask)

    def between_lines_masker(masks_l0,masks_l1):
        # check up-down
        masked = masks_l0[1] & masks_l1[0]
        if len(masked[masked>0]) == 0:
            masked = masks_l0[0] & masks_l1[1]
        return masked

    def ga_config(parcels):
        return pygad.GA(num_generations=5,
                       num_parents_mating=2,
                    #    initial_population=[np.arange(0.2, 1, 1/parcels) for _ in range(10)],
                       fitness_func=Parcels.fitness_func,
                       sol_per_pop=10,
                       num_genes=parcels-1,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       random_mutation_max_val=0.5,
                       random_mutation_min_val=-0.5,
                       gene_space=list(np.arange(0.2, 0.8, 0.05)),
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type=None,
                       mutation_type="random",
                       allow_duplicate_genes=True,
                       mutation_percent_genes=100)
    """
        decode fitness function
        each gene contains the relative selection 
    """
    def decoder(solution):
        # convert solution to points & sort them
        max_indx = len(Parcels.axis_line_arr)
        solution = np.asarray(solution)
        solution = solution*max_indx
        solution = solution.astype(np.uint16)
        solution = np.sort(solution)
        points = np.asarray(Parcels.axis_line_arr)[solution]
        # add first and end point of axis line
        points = list(points)
        points.insert(0,Parcels.axis_line_arr[0])
        points.append(Parcels.axis_line_arr[-1])
        # calculate a line pp to axis on each points
        out_points_f = [] # front
        out_points_b = [] # back
        # axis line
        y1, x1 = Parcels.axis_line_arr[0]
        y2, x2 = Parcels.axis_line_arr[-1]
        p1, p2 = Point(x1, y1), Point(x2, y2)
        l1 = Line(p1, p2)
        for point in points:
            l2 = l1.perpendicular_line((point[1],point[0]))
            cr = Circle((point[1],point[0]),Parcels.circle_line_len)
            list_p0 = list(cr.intersect(l2).args[0])
            list_p1 = list(cr.intersect(l2).args[1])
            out_points_f.append((int(list_p0[1]),int(list_p0[0])))
            out_points_b.append((int(list_p1[1]),int(list_p1[0])))
        return list(zip(out_points_f,out_points_b))

    """
        calculates fixed facility hit
        best answer is 1
    """
    def cal_fixed_facilities_fitness(facility_mask:np.ndarray,line_mask:np.ndarray):
        imgray = facility_mask
        collision = line_mask.astype(np.uint8) & imgray
        max_collision = np.sum(imgray[imgray>0])
        if max_collision == 0: return 1
        collision = np.sum(collision[collision>0])
        return 1 - (collision/max_collision)
    """
        calculates cut trees with given points
        no cut tree = 1
    """
    def cal_carbon_fitness(tree_mask:np.ndarray,line_mask:np.ndarray):
        mask = tree_mask
        collision = line_mask.astype(np.uint8) & mask
        # frame must be preprocessed
        max_carbon = np.sum(mask)
        if max_carbon == 0: return (1,0)
        return (1-(np.sum(mask[collision>0])/max_carbon),np.sum(mask[collision>0]))
    
    def cal_area(lines_points_tup):
        parcel_areas = []
        for two_lines in lines_points_tup:
            img = np.zeros(Parcels.map.block_mask.shape)
            p0 = (two_lines[0][0][1],two_lines[0][0][0])
            p1 = (two_lines[0][1][1],two_lines[0][1][0])
            cv2.line(img,p0,p1,255,2)
            masks_l0 = Parcels.line_split_maker(img,two_lines[0][0],two_lines[0][1])
            img = np.zeros(Parcels.map.block_mask.shape)
            p0 = (two_lines[1][0][1],two_lines[1][0][0])
            p1 = (two_lines[1][1][1],two_lines[1][1][0])
            cv2.line(img,p0,p1,255,2)
            masks_l1 = Parcels.line_split_maker(img,two_lines[1][0],two_lines[1][1])
            masked = Parcels.between_lines_masker(masks_l0,masks_l1)
            # apply block mask
            masked = masked & Parcels.map.block_mask
            # cal area of block
            parcel_areas.append(np.sum(masked[masked>0])/255)
        return parcel_areas
    
    def cal_area_fitness(parcels_area:list):
        area_score = 0
        has_types = [0,0,0]
        for parcel in parcels_area:
            if parcel < config.TYPE_C_AREA - config.TYPE_AREA_STEP:
                area_score -= 3
            elif parcel <= config.TYPE_C_AREA:
                has_types[0] += 1
            elif parcel <= config.TYPE_B_AREA:
                has_types[1] += 1
            elif parcel <= config.TYPE_A_AREA:
                has_types[2] += 1
            elif parcel > config.TYPE_A_AREA:
                area_score -= 0.25
        
        for i in has_types:
            if i > 0:
                area_score += 0.33*(i-1)

        return area_score
    
    def has_collision(solution):
        sol = np.sort(solution)
        for i,elm in enumerate(sol):
            poped_sol = np.delete(sol,i)
            if (np.absolute(poped_sol - elm) <= 0.2).any(): return True
        return False
    """
        fitness function for partitioning
        solution is arr range [0,1] of size parcels count
        arr[0] selects the first line to be drawn
        arr[1:] select how much furthe next lines 
        are from first line
    """
    def fitness_func(solution,solution_idx):
        # collision check hard condition
        if Parcels.has_collision(solution): return -1
        # decode arr
        lines_points = Parcels.decoder(solution)
        # cal parcels split masks
        lines_points_tup = list(zip(lines_points[::2],lines_points[1::2]))
        lines_points_tup.extend(list(zip(lines_points[1::2],lines_points[2::2])))
        lines_points_tup.sort(key=lambda x: x[0][0])
        # cal parcels area
        parcels_area = Parcels.cal_area(lines_points_tup)
        # all draw lines
        plain = np.zeros((Parcels.img_shape))
        # input points are (y,x)->(x,y)
        lines_points = lines_points[1:-1]
        for line in lines_points:
            plain = cv2.line(plain,(line[0][1],line[0][0]),(line[1][1],line[1][0]),255,config.FACILITY_SAFE_DIST)
        # pass mask to fitness calculators
        tree_fitness = Parcels.cal_carbon_fitness(Parcels.map.trees_mask,plain)
        ff_fitness = Parcels.cal_fixed_facilities_fitness(Parcels.map.fixed_f_mask,plain)
        # score
        # area score
        area_score = Parcels.cal_area_fitness(parcels_area)
        if ff_fitness != 1:
            ff_fitness = -1*ff_fitness
        
        weights = config.P_AREA_WEIGHT + config.P_FF_WEIGHT + config.P_TREE_WEIGHT
        score = config.P_FF_WEIGHT*ff_fitness + config.P_AREA_WEIGHT *area_score + config.P_TREE_WEIGHT*tree_fitness[0]
        return score/weights
 
    def map_maker(solution,p_cnt):
        if p_cnt != 1:
            lines_points = Parcels.decoder(solution)
            # cal parcels split masks
            lines_points_tup = list(zip(lines_points[::2],lines_points[1::2]))
            lines_points_tup.extend(list(zip(lines_points[1::2],lines_points[2::2])))
            lines_points_tup.sort(key=lambda x: x[0][0])
        else:
            points = []
            points.insert(0,Parcels.axis_line_arr[0])
            points.append(Parcels.axis_line_arr[-1])
            # calculate a line pp to axis on each points
            out_points = []
            # axis line
            y1, x1 = Parcels.axis_line_arr[0]
            y2, x2 = Parcels.axis_line_arr[-1]
            p1, p2 = Point(x1, y1), Point(x2, y2)
            l1 = Line(p1, p2)
            for point in points:
                l2 = l1.perpendicular_line((point[1],point[0]))
                cr = Circle((point[1],point[0]),Parcels.circle_line_len)
                list_p = list(cr.intersect(l2).args[Parcels.map.dir])
                out_points.append((int(list_p[1]),int(list_p[0])))
            lines_points = list(zip(out_points,points))
            lines_points_tup = list(zip(lines_points[::2],lines_points[1::2]))
            lines_points_tup.extend(list(zip(lines_points[1::2],lines_points[2::2])))
            lines_points_tup.sort(key=lambda x: x[0][0])
        # cal parcels area
        parcel_areas = []
        parcel_masks = []
        for two_lines in lines_points_tup:
            img = np.zeros(Parcels.map.block_mask.shape)
            p0 = (two_lines[0][0][1],two_lines[0][0][0])
            p1 = (two_lines[0][1][1],two_lines[0][1][0])
            cv2.line(img,p0,p1,255,2)
            masks_l0 = Parcels.line_split_maker(img,two_lines[0][0],two_lines[0][1])
            img = np.zeros(Parcels.map.block_mask.shape)
            p0 = (two_lines[1][0][1],two_lines[1][0][0])
            p1 = (two_lines[1][1][1],two_lines[1][1][0])
            cv2.line(img,p0,p1,255,2)
            masks_l1 = Parcels.line_split_maker(img,two_lines[1][0],two_lines[1][1])
            masked = Parcels.between_lines_masker(masks_l0,masks_l1)
            # apply block mask
            masked = masked & Parcels.map.block_mask
            parcel_masks.append(masked)
            # cal area of block
            parcel_areas.append(len(masked[masked>0]))
        
        # make new maps
        report_dic = {p_type:0 for p_type in config.ParcelType._member_names_}
        report_dic['cnt'] = p_cnt
        parcels_maps = []
        for indx,split_mask in enumerate(parcel_masks):
            parcel_type = 0
            if parcel_areas[indx] < config.TYPE_C_AREA - config.TYPE_AREA_STEP:
                parcel_type = config.ParcelType.U
                report_dic['U'] += 1
            elif parcel_areas[indx] <= config.TYPE_C_AREA:
                parcel_type = config.ParcelType.C
                report_dic['C'] += 1
            elif parcel_areas[indx] <= config.TYPE_B_AREA:
                parcel_type = config.ParcelType.B
                report_dic['B'] += 1
            elif parcel_areas[indx] <= config.TYPE_A_AREA:
                parcel_type = config.ParcelType.A
                report_dic['A'] += 1
            else:
                parcel_type = config.ParcelType.O
                report_dic['O'] += 1
            parcels_maps.append(MapIn(indx,split_mask,Parcels.map,parcel_areas[indx],lines_points_tup[indx],parcel_type))
        
        return (parcels_maps,report_dic)

    def estimate_parcel_num(parcels,area):
        p_types = {p_type:0 for p_type in config.ParcelType._member_names_}
        parcels_cnt = 0
        while area > config.TYPE_B_AREA - config.TYPE_AREA_STEP:
            if area - (config.TYPE_B_AREA - config.TYPE_AREA_STEP/2) > 0:
                parcels_cnt += 1
                p_types['B'] += 1
                area -= config.TYPE_B_AREA
            if area - (config.TYPE_A_AREA - config.TYPE_AREA_STEP/2) > 0:
                parcels_cnt += 1
                p_types['A'] += 1
                area -= config.TYPE_A_AREA
            if area - (config.TYPE_C_AREA - config.TYPE_AREA_STEP/2) > 0:
                parcels_cnt += 1
                p_types['C'] += 1
                area -= config.TYPE_C_AREA
        return (max(parcels,parcels_cnt),p_types)

    def ga_run(self,threshold = config.P_GA_LOSS_THOLD):
        score = threshold - 1
        # parcel estimator
        parcels,p_types = Parcels.estimate_parcel_num(self.parcels,self.map.curr_size)
        config.log(f"Map {Parcels.map.map_id} suggested parcel num:{parcels}")
        config.log(f"Map {Parcels.map.map_id} suggested parcel types:{p_types}")
        # parcels area check
        while(score < threshold):
            if parcels < 2:
                self.best_points = None
                self.best_solution = 'Nothing To Split!'
                score = 1
            else:
                self.ga_instance = Parcels.ga_config(parcels)
                self.ga_instance.run()
                # solution, solution_fitness, solution_idx
                self.best_solution = self.ga_instance.best_solution()
                self.best_points = Parcels.decoder(self.best_solution[0])[1:-1]
                # self.parcels = Parcels.make_map(self.best_solution)
                score = self.best_solution[1]
            parcels -= 1
        
        if (parcels+1 != self.parcels):
            config.log(f"Map {Parcels.map.map_id} Has \"{self.parcels-parcels-1}\" Difference Parcel Count Due To Uncertainity")
        # make maps
        self.parcels_maps, self.report = Parcels.map_maker(self.best_solution[0],parcels+1)
        self.map = copy.deepcopy(Parcels.map)

        return self
