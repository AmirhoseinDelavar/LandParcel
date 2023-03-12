import cv2
import numpy as np
from Map import MapIn, CVLineThickness
import random
import config

class AxisFinder:
    clicked_points = []
    def __init__(self,map:MapIn) -> None:
        self.map = map
        self.axis_res = []
        
    """
        opencv click callback for customized axis drawing
    """
    def click_callback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            config.log(f"Clicked Y:{y}, X:{x}")
            AxisFinder.clicked_points.append((y,x))

    """
        parts parcel number divider based on area of
        parts. omits fixed facility areas
    """
    def cal_split_parcels(u_map:MapIn,d_map:MapIn,parcels_cnt:int):
        u_ff_area = np.sum(u_map.facility_filled_mask)/255
        u_block_area = np.sum(u_map.block_mask)/255
        u_area = u_block_area - u_ff_area
        d_ff_area = np.sum(d_map.facility_filled_mask)/255
        d_block_area = np.sum(d_map.block_mask)/255
        d_area = d_block_area - d_ff_area
        t_area = u_area+d_area

        precnt = u_area/t_area
        u_parcels = int(parcels_cnt*precnt)
        d_parcels = parcels_cnt - u_parcels
        return (u_parcels,d_parcels)
    """
        sort results by best fitness
    """
    def sort_fitness(self,sub_li):
            return sub_li.sort(key = lambda x: x[0])

    """
        calculates access balance ratio with the provided up&down masks
        max value of fitness is 1
    """
    def cal_access_split_fitness(self,access_mask:np.ndarray,up_mask:np.ndarray,down_mask:np.ndarray):
        img = access_mask
        # calculate access ratio
        up_sum_mask = up_mask & img
        down_sum_mask = down_mask & img

        sum_up_access = np.sum(up_sum_mask)/255
        sum_down_access = np.sum(down_sum_mask)/255
        return 1 - (abs(sum_up_access-sum_down_access)/(np.sum(img)/255))
    """
        calculates area balance ratio with provided up&down masks
        max value of fitness is 1
    """
    def cal_area_split_fitness(self,block_mask:np.ndarray,up_mask:np.ndarray,down_mask:np.ndarray):
        imgray = block_mask
        up_area = up_mask & imgray
        down_area = down_mask & imgray
        sum_up_area = np.sum(up_area)/255
        sum_down_area = np.sum(down_area)/255
        return 1 - (abs(sum_up_area-sum_down_area)/(np.sum(imgray)/255))
    """
        calculates fixed facility hit
        best answer is 1
    """
    def cal_fixed_facilities_fitness(self,facility_mask:np.ndarray,p0:tuple,p1:tuple):
        imgray = facility_mask
        plain = np.zeros((imgray.shape))
        thickness = self.map.roud_thickness + self.map.facility_safe_dist
        # input points are (y,x)->(x,y)
        plain = cv2.line(plain,(p0[1],p0[0]),(p1[1],p1[0]),255,CVLineThickness.thickness_solver(thickness))
        collision = plain.astype(np.uint8) & imgray
        max_collision = np.sum(imgray)/255
        if max_collision == 0: return 1
        collision = np.sum(collision)/255
        return 1 - (collision/max_collision)
    """
        calculates cut trees with given points
        no cut tree = 1
    """
    def cal_carbon_fitness(self,tree_mask:np.ndarray,p0:tuple,p1:tuple):
        mask = tree_mask
        plain = np.zeros((mask.shape))
        thickness = self.map.roud_thickness + self.map.tree_safe_dist
        # input points are (y,x)->(x,y)
        plain = cv2.line(plain,(p0[1],p0[0]),(p1[1],p1[0]),255,CVLineThickness.thickness_solver(thickness))
        collision = plain.astype(np.uint8) & mask
        # frame must be preprocessed
        max_carbon = np.sum(mask)
        if max_carbon == 0: return (1,0)
        return (1-(np.sum(mask[collision>0])/max_carbon),len(mask[collision>0]))
    """
        axis finding fitness function
    """
    def fitness_axis(self,solution:tuple,mymap:MapIn,center:tuple):
        # solution is y,x of one point
        # other point is the center
        y_max = mymap.frame_shape[0]-1
        x_max = mymap.frame_shape[1]-1
        # slope = inf, x=const
        if solution[1]==center[1]:
            point0 = (0,center[1])
            point1 = (y_max,center[1])
        # slope = zero, y=const
        elif solution[0]==center[0]:
            point0 = (center[0],0)
            point1 = (center[0],x_max)
        # normal line
        else:
            slope = (solution[0]-center[0])/(solution[1]-center[1]) 
            intercept = center[0] - (slope*center[1])
            point0 = (int(intercept),0)
            point1 = (int((slope*x_max)+intercept),x_max)

        # config.log(point1,point0)
        up_mask,down_mask = mymap.line_split_mask_maker(point0,point1)
        access_split_fitness = self.cal_access_split_fitness(mymap.access_mask,up_mask,down_mask)
        area_split_fitness = self.cal_area_split_fitness(mymap.block_mask,up_mask,down_mask)
        fixed_facilities_fitness = self.cal_fixed_facilities_fitness(mymap.fixed_f_mask,point0,point1)
        carbon_fitness = self.cal_carbon_fitness(mymap.trees_mask,point0,point1)

        # config.log(solution,slope,intercept,point0,point1)
        weights = config.A_ACCESS_SPLIT_WEIGHT + config.A_AREA_SPLIT_WEIGHT + config.A_FIXED_FACILITIES_WEIGHT + config.A_CARBON_WEIGHT
        score = (config.A_ACCESS_SPLIT_WEIGHT * access_split_fitness + config.A_AREA_SPLIT_WEIGHT * area_split_fitness + 
         config.A_FIXED_FACILITIES_WEIGHT * fixed_facilities_fitness + config.A_CARBON_WEIGHT * carbon_fitness[0])

        # scale score between (0,1]
        if fixed_facilities_fitness != 1:
            return (-1,point0,point1)
        return (score/weights,point0,point1,solution,center,[access_split_fitness,area_split_fitness,fixed_facilities_fitness,carbon_fitness])
    """
        iterate through all border pixels and
        draw line on start_point to border
        finding best line (for the first step of axis finding only)
    """
    def iterate_throughall(self,mymap:MapIn, start_point=None):
        borders_access = []
        if start_point != None:
            borders_access = [start_point]
        else:
            borders_access = mymap.access_mask
            borders_access = np.asarray(np.where(borders_access==255))
            borders_access = list(zip(borders_access[0], borders_access[1]))
        
        borders_random = mymap.boundry_mask
        borders_random = np.asarray(np.where(borders_random==255))
        borders_random = list(zip(borders_random[0], borders_random[1]))
        borders_random = random.sample(borders_random, int(len(borders_random)*0.4))

        res = []
        for pixel in borders_access:
            for second_point in borders_random:
                if np.linalg.norm(np.asarray(pixel)-np.asarray(second_point)) > config.VALID_POINT_DISTANCE:
                    res.append(self.fitness_axis(pixel,mymap,second_point))
        self.sort_fitness(res)
        self.axis_res = res[-10:]
        return res[-10:]
    
    """
        iterate over old boundries (without division line)
        and find best line for New York design
    """
    def iterate_old_boundries_new_york(self):
        res = []
        last_axis_points = self.map.line_points
        boundary = self.map.old_boundry_mask
        boundary = np.asarray(np.where(boundary==255))
        boundary = list(zip(boundary[0], boundary[1]))
        # cal perpendicular center (pixelhit line)
        for pixel in boundary:
            y1, x1 = last_axis_points[0]
            y2, x2 = last_axis_points[1] 
            y3, x3 = pixel
            px, py = (x2-x1,y2-y1)
            dAB = px*px + py*py
            u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
            # (y,x)
            ppcenter = (int(y1 + u * py),int(x1 + u * px))
            
            if np.linalg.norm(np.asarray(pixel)-np.asarray(ppcenter)) > config.VALID_POINT_DISTANCE:
                res.append(self.fitness_axis(pixel,self.map,ppcenter))
        
        self.sort_fitness(res)
        self.axis_res = res[-10:]
        return res[-10:]
