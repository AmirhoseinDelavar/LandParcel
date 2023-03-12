import cv2
import numpy as np
import pandas as pd
import random
from functools import singledispatchmethod
import math
import os

from sympy import false
import config

class MapIn:
    # thickness configs
    tree_safe_dist = config.TREE_SAFE_DIST
    facility_safe_dist = config.FACILITY_SAFE_DIST
    # parcel min area
    parcel_minimum_area = config.AXIS_MIN_AREA
    # ratio of access to boundry to stop halving
    access_ratio = config.ACCESS_RATIO

    """
        map class to handle raw inputs
        RGB (round access, green factor, boundary)
        Background is white by deafult
        Black shows fixed facilities
    """
    @singledispatchmethod
    def __init__(self) -> None:
        assert False, 'bad input'
    @__init__.register(str)
    def _first__(self,src:str,src_block:str,src_ff:str,parcel_cnt:int,arch_choice:config.ArchStyles,map_id:int) -> None:
        self.roud_thickness = config.ROAD_SIZE_MAX
        self.map_id = map_id
        self.frame = cv2.imread(src)
        self.frame_shape = self.frame.shape
        self.arch_choice = arch_choice
        self.parcel_cnt = parcel_cnt
        self.centers = (int(self.frame_shape[0]/2),int(self.frame_shape[1]/2))
        self.trees_mask, self.fixed_f_mask,self.access_mask, self.boundry_mask = self.create_masks()
        self.trees_binary_mask = cv2.threshold(self.trees_mask, 127, 255, cv2.THRESH_BINARY)[1]
        # read block mask
        self.block_mask = cv2.imread(src_block)
        self.block_mask = cv2.cvtColor(self.block_mask,cv2.COLOR_BGR2GRAY)
        # read facility filled mask
        self.facility_filled_mask = cv2.imread(src_ff)
        self.facility_filled_mask = cv2.cvtColor(self.facility_filled_mask,cv2.COLOR_BGR2GRAY)
        self.print_report()
        
    # Axis maps initialization
    @__init__.register(np.ndarray)
    def _second__(self,split_mask:np.ndarray,parent_map,line_mask:np.ndarray,map_id:int,dir:int,line_p) -> None:
        self.line_p = line_p
        # self.axis_center = parent_map.axis_center
        self.split_mask = split_mask
        self.dir = dir # 0 up 1 down
        self.roud_thickness = config.ROAD_SIZE_MAX - config.ROAD_STEP*int(math.log(map_id+1,2))
        if self.roud_thickness <= config.ROAD_STEP: self.roud_thickness=config.ROAD_SIZE_MIN
        config.log(f"roud thickness:{self.roud_thickness} map_id:{map_id}")
        self.map_id = map_id
        # split_mask_3d = np.zeros((self.frame_shape))
        # self.frame = parent_map.frame & split_mask
        split_3d_mask = np.zeros(parent_map.frame_shape, dtype=np.uint8)
        split_3d_mask[:,:,:] = split_mask[:,:,np.newaxis]
        self.frame = parent_map.frame & split_3d_mask
        self.frame_shape = self.frame.shape
        self.arch_choice = parent_map.arch_choice
        self.centers = (int(self.frame_shape[0]/2),int(self.frame_shape[1]/2))
        self.trees_mask = parent_map.trees_mask & split_mask
        self.trees_binary_mask = parent_map.trees_binary_mask & split_mask
        self.fixed_f_mask = parent_map.fixed_f_mask & split_mask
        self.boundry_mask = parent_map.boundry_mask & split_mask
        self.old_boundry_mask = parent_map.boundry_mask & split_mask
        self.block_mask = parent_map.block_mask & split_mask
        self.facility_filled_mask = parent_map.facility_filled_mask & split_mask
        # add new line to access
        new_access_line = self.block_mask & line_mask
        self.access_mask = parent_map.access_mask & split_mask
        self.access_mask = self.access_mask | new_access_line
        # add new line to boundries
        self.boundry_mask = self.boundry_mask | new_access_line
        self.new_access_line = new_access_line
        if map_id > 2:
            self.parent_access_line = parent_map.new_access_line
            self.parent_line_p = parent_map.line_p
        else:
            self.parent_access_line = self.new_access_line
            self.parent_line_p = self.line_p
        self.save_map()
        self.print_report()
    
    # Parcels initilization
    @__init__.register(int)
    def _third__(self,parcel_id:int,split_mask:np.ndarray,parent_map,parcel_area,lines_points_tup,parcel_type) -> None:
        self.dir = parent_map.dir
        self.parent_line_p = parent_map.parent_line_p
        self.line_p = parent_map.line_p
        self.parent_access_line = parent_map.parent_access_line & split_mask
        self.access_line = parent_map.new_access_line & split_mask
        self.parcel_id = parcel_id
        self.curr_size = parcel_area
        self.bounding_lines = lines_points_tup
        self.parcel_type = parcel_type
        self.map_id = parent_map.map_id
        split_3d_mask = np.zeros(parent_map.frame_shape, dtype=np.uint8)
        split_3d_mask[:,:,:] = split_mask[:,:,np.newaxis]
        self.frame = parent_map.frame & split_3d_mask
        self.frame_shape = self.frame.shape
        self.arch_choice = parent_map.arch_choice
        self.trees_mask = parent_map.trees_mask & split_mask
        self.trees_binary_mask = parent_map.trees_binary_mask & split_mask
        self.fixed_f_mask = parent_map.fixed_f_mask & split_mask
        self.boundry_mask = parent_map.boundry_mask & split_mask
        self.block_mask = parent_map.block_mask & split_mask
        self.facility_filled_mask = parent_map.facility_filled_mask & split_mask
        # make center of new parcel
        block_mask = self.block_mask.astype(np.uint8)
        contours, _ = cv2.findContours(block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        M = cv2.moments(cnts[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self.parcel_center = (cY,cX)
        # self.save_map()
        self.print_report()

    def print_report(self):
        config.log(f"Map {self.map_id} Area : {np.sum(self.block_mask)/255}")
        config.log(f"Map {self.map_id} Tree Area : {np.sum(self.trees_binary_mask)/255}")
        config.log(f"Map {self.map_id} Fixed-Facility Area : {np.sum(self.facility_filled_mask)/255}")
        config.log(f"Map {self.map_id} Sparse Area : {np.sum(self.block_mask & np.bitwise_not(self.facility_filled_mask) & np.bitwise_not(self.trees_binary_mask))/255}")

    def set_map_axis_center(self,point):
        self.axis_center = point
    def set_line_point(self,point:tuple):
        self.line_points = point

    def save_map(self) -> None:
        if config.WRITE_UNNECESSARY:
            cv2.imwrite(f'outputs/map{self.map_id}.bmp',self.frame)
            cv2.imwrite(f'outputs/access_mask{self.map_id}.bmp',self.access_mask)
            cv2.imwrite(f'outputs/boundry_mask{self.map_id}.bmp',self.boundry_mask)

    def create_masks(self) -> tuple:
        res = []
        img_re = self.frame.reshape(-1,3)
        df = pd.DataFrame(img_re,columns=['b','g','r'])
        df['r'].astype(np.uint8)
        df['g'].astype(np.uint8)
        df['b'].astype(np.uint8)

        indx_trees = df.apply(lambda x: x.b==0 and 0<x.g<=255 and x.r==0, axis=1)
        df_trees = df.copy()
        df_trees[np.logical_not(indx_trees)] = [0,0,0]
        # df_trees[indx_trees] = [255,255,255]
        out = df_trees.values.reshape(self.frame_shape)
        out = out.astype(np.uint8)
        out[:,:,0] = 0
        out[:,:,2] = 0
        out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)[1][:,:,1]
        res.append(out)
        cv2.imwrite('outputs/tree_mask.bmp',out)

        indx_fixed_fac = df.apply(lambda x: x.b==0 and x.g==0 and x.r==0, axis=1)
        df_ff = df.copy()
        df_ff[np.logical_not(indx_fixed_fac)] = [0,0,0]
        df_ff[indx_fixed_fac] = [255,255,255]
        out = df_ff.values.reshape(self.frame_shape)
        out = out.astype(np.uint8)
        out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
        res.append(out)
        cv2.imwrite('outputs/facility_mask.bmp',out)

        indx_access = df.apply(lambda x: x.g==0 and x.r==255, axis=1)
        df_ac = df.copy()
        df_ac[np.logical_not(indx_access)] = [0,0,0]
        df_ac[indx_access] = [255,255,255]
        out = df_ac.values.reshape(self.frame_shape)
        out = out.astype(np.uint8)
        out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
        res.append(out)
        cv2.imwrite('outputs/access_mask.bmp',out)

        indx_boundry = df.apply(lambda x: x.b==255 and x.g==0, axis=1)
        df_b = df.copy()
        df_b[np.logical_not(indx_boundry)] = [0,0,0]
        df_b[indx_boundry] = [255,255,255]
        out = df_b.values.reshape(self.frame_shape)
        out = out.astype(np.uint8)
        out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
        res.append(out)
        cv2.imwrite('outputs/boundary_mask.bmp',out)
        return tuple(res)

    def correct_input(self) -> None:
        img_re = self.frame.reshape(-1,3)
        df = pd.DataFrame(img_re,columns=['b','g','r'])
        df['r'].astype(np.uint8)
        df['g'].astype(np.uint8)
        df['b'].astype(np.uint8)
        # ----set tree values to random
        random.seed(13)
        df = df.apply(lambda x: [0,random.randint(1,255),0] if x['b']==0 and x['g']==255 and x['r']==0 else x,axis=1)
        # ----convert white to black
        # indx = df.apply(lambda x: x['b']==255 and x['g']==255 and x['r']==255,axis=1)
        # df[indx] = [0,0,0]
        # ----set rgb(0,255,255) to rgb(255,0,255)
        # indx = df.apply(lambda x: x['b']==255 and x['g']==255 and x['r']==0,axis=1)
        # df[indx] = [255,0,255]
        # ----save output
        out = df.values.reshape(self.frame_shape)
        out = out.astype(np.uint8)
        cv2.imwrite('outputs/kan_pre.bmp',out)
        self.frame = out
        # ----print
        # cv2.imshow("kan_pre_out", out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    """
        returns above the line mask and below the line mask
    """
    def line_split_mask_maker(self,p0:tuple,p1:tuple):
        # points are (y,x) oriented
        img_pixels = self.frame_shape[0]*self.frame_shape[1]
        img_x = self.frame_shape[1]
        # numpy image is (y*x*3)
        # !! unit16 may not be enough
        y_index = (np.arange(img_pixels).reshape(self.frame_shape[:2])/img_x).astype(np.uint32)
        x_index = np.arange(img_pixels).reshape(self.frame_shape[:2])%img_x
        if p1[1] == p0[1]:
            up_down_line = x_index - p0[1]
        else:
            slope = (p1[0]-p0[0])/(p1[1]-p0[1]) 
            intercept = p0[0] - (slope*p0[1])
            up_down_line = x_index*slope + intercept - y_index
        # down part (becareful about center)
        down_mask = np.where(up_down_line>=0,255,0).reshape(self.frame_shape[:2])
        up_mask = np.where(up_down_line>=0,0,255).reshape(self.frame_shape[:2])
        return (up_mask,down_mask)
    """
        returns only line mask on main image
    """
    def line_mask_maker(self,p0:tuple,p1:tuple):
        plain = np.zeros((self.block_mask.shape))
        plain = cv2.line(plain,(p0[1],p0[0]),(p1[1],p1[0]),255,2)
        return plain.astype(np.uint8)

    
    """
        check whether the half map has a feasible condition 
        or supports the finishing condtion.
    """
    def isfeasible(self):
        # check if the part has more than 60% access
        access = np.sum(self.access_mask)/255
        boundry = np.sum(self.boundry_mask)/255
        access_ratio = access/boundry
        access_cond = access_ratio<self.access_ratio
        # area of part not smaller than standard
        block_size = np.sum(self.block_mask)/255
        size_cond = block_size>self.parcel_minimum_area
        config.log(f'block size:{block_size} access_ratio:{access_ratio} map_id:{self.map_id}')
        self.curr_access = access_ratio
        self.curr_size = block_size
        return access_cond and size_cond

class CVLineThickness:
    """
        method selects cv2line arg 
        depending on the pixel width
    """
    @staticmethod
    def thickness_solver(desired_thickness):
        if desired_thickness == 1:
            return 1
        if desired_thickness == 2:
            # assert false, f"change road size or road step to odd number: {desired_thickness}"
            return 2
        if desired_thickness == 3:
            return 2
        if desired_thickness % 2 == 0:
            # assert false, f"change road size or road step to odd number: {desired_thickness}"
            return desired_thickness - 1
        
        return desired_thickness - 2


class MapOut:
    def __init__(self,src:str,lines_axis:list) -> None:
        self.img = cv2.imread(src)
        self.img_axised = self.img.copy()
        self.img_partitioned = None
        self.img_built = None
        self.img_last = None
        self.axis_lines = lines_axis
        self.partitioning_lines = None
        self.parcels_dic = {}
        self.building_masks = None
        # export details
        self.total_carbon = 0
        self.total_trees = 0
        self.total_carbon_loss = 0
        self.total_cut_tree = 0
        self.total_axis_length = 0
        self.total_axis_per_block_pr = 0
        self.total_num_parcels = 0
        self.total_num_parcels_types = {p_type:0 for p_type in config.ParcelType._member_names_}
        self.total_sum_ff = 0
    
    def reset_map_for_partitioning(self):
        self.total_num_parcels = 0
        self.total_num_parcels_types = {p_type:0 for p_type in config.ParcelType._member_names_}
        self.total_sum_ff = 0
        self.partitioning_lines = None
        if self.img_partitioned is not None:
            self.img_partitioned = None    
        self.img_last = self.img_axised.copy()

    def reset_map_for_location_finding(self):
        self.total_sum_ff = 0
        self.building_masks = None
        if self.img_built is not None:
            self.img_built = None
        self.img_last = self.img_partitioned.copy()
    
    def add_partition_report(self,report):
        self.total_num_parcels += report['cnt']
        report.pop('cnt')
        for p_type in report.keys():
            self.total_num_parcels_types[p_type] += report[p_type]

    def report(self):
        self.block_mask = cv2.imread(config.MAIN_MAP_FILLED_BLOCK_MASK)
        self.tree_mask = cv2.imread('outputs/tree_mask.bmp')
        self.binary_tree_mask = cv2.threshold(self.tree_mask, 127, 255, cv2.THRESH_BINARY)[1]
        self.facility_filled_mask = cv2.imread(config.MAIN_MAP_FILLED_F_F_MASK)
        # total trees carbon
        self.total_carbon = np.sum(self.tree_mask)/3
        self.total_trees = np.sum(self.binary_tree_mask)/(255*3)
        # calculate tree cut and carbon
        self.img_last = self.img_last & self.block_mask
        self.img_last = self.img_last.astype(np.uint8)
        self.img_mask = cv2.threshold(self.img_last, 127, 255, cv2.THRESH_BINARY)[1]
        # omit roads mask and building mask from it
        self.roads_mask = cv2.imread('outputs/roads_mask.bmp')
        collision3dmask = self.roads_mask
        if os.path.exists('outputs/buildings_mask.bmp'):
            collision3dmask = collision3dmask | cv2.imread('outputs/buildings_mask.bmp')
        if os.path.exists('outputs/partitioning_mask.bmp'):
            collision3dmask = collision3dmask | cv2.imread('outputs/partitioning_mask.bmp')
        cv2.imwrite('outputs/constructed_mask.bmp', collision3dmask)
        self.total_carbon_loss = np.sum(collision3dmask & self.tree_mask)/3
        self.total_cut_tree = np.sum(collision3dmask & self.binary_tree_mask)/(255*3)
        config.log(f"Total Trees:{self.total_trees} Total Carbon Values:{self.total_carbon}")
        config.log(f"Total Cut Trees:{self.total_cut_tree} Total Carbon Loss:{self.total_carbon_loss}")
        config.log(f"Total Cut Precentage:{self.total_cut_tree/self.total_trees}")
        # calculate axis reports
        img_re = self.img_last.reshape(-1,3)
        df = pd.DataFrame(img_re,columns=['b','g','r'])
        df['r'].astype(np.uint8)
        df['g'].astype(np.uint8)
        df['b'].astype(np.uint8)
        indx_axis = df.apply(lambda x:x.g == 0 and 0<x.r<=255 and x.b == 0, axis=1)
        df_axis = df.copy()
        df_axis[np.logical_not(indx_axis)] = [0,0,0]
        out = df_axis.values.reshape(self.img_last.shape)
        out = out.astype(np.uint8)
        out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
        out = cv2.threshold(out, 1, 255, cv2.THRESH_BINARY)[1]
        self.total_axis_length = np.sum(out)/255
        self.total_axis_per_block_pr = self.total_axis_length / (np.sum(self.block_mask)/(255*3))
        sparse_area = np.sum(self.block_mask & np.bitwise_not(self.facility_filled_mask) & np.bitwise_not(self.binary_tree_mask))/(255*3)
        self.total_axis_per_sparse_pr = self.total_axis_length / sparse_area
        config.log(f"Total Axis Area:{self.total_axis_length}")
        config.log(f"Total Block Area:{np.sum(self.block_mask)/(255*3)}")
        config.log(f"Total Sparse Area:{sparse_area}")
        config.log(f"Total Axis Per Block Precentage:{self.total_axis_per_block_pr}")
        config.log(f"Total Axis Per Sparse Precentage:{self.total_axis_per_sparse_pr}")
        # Parcels number plus types
        config.log(f"Total Parcels:{self.total_num_parcels}")
        config.log(f"Total Parcel types:{self.total_num_parcels_types}")
        config.log(f"Total Parcels With FF:{self.total_sum_ff}")


        
    def draw_axis(self):
        for line in self.axis_lines:
            p0=line[0][0]
            p1=line[0][1]
            thickness=config.ROAD_SIZE_MAX - config.ROAD_STEP*int(math.log(line[1]+1,2))
            if thickness <= config.ROAD_SIZE_MIN:
                thickness=config.ROAD_SIZE_MIN
            # draw line of axis
            self.img_axised = cv2.line(self.img_axised,(p0[1],p0[0]),(p1[1],p1[0]),(0,0,127),CVLineThickness.thickness_solver(thickness+2))
            # rotate points
            self.img_axised = cv2.line(self.img_axised,(p0[1],p0[0]),(p1[1],p1[0]),(0,0,255),CVLineThickness.thickness_solver(thickness))
        cv2.imwrite('outputs/final_axis.bmp',self.img_axised)
        # save road lines mask
        img_re = self.img_axised.reshape(-1,3).copy()
        df = pd.DataFrame(img_re,columns=['b','g','r'])
        df['r'].astype(np.uint8)
        df['g'].astype(np.uint8)
        df['b'].astype(np.uint8)
        indx_axis = df.apply(lambda x:x.g == 0 and 0<x.r<=255 and x.b == 0, axis=1)
        df[np.logical_not(indx_axis)] = [0,0,0]
        out = df.values.reshape(self.img_axised.shape)
        out = out.astype(np.uint8)
        out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
        out = cv2.threshold(out, 1, 255, cv2.THRESH_BINARY)[1]
        self.img_last = self.img_axised.copy()
        cv2.imwrite('outputs/roads_mask.bmp', out)
    
    def draw_partitions(self,iteration:int,map:MapIn,lines_parcels:list):
        map_id = map.map_id
        if lines_parcels is not None:
            self.parcels_dic[map_id] = lines_parcels
            lines_mask = np.zeros(self.img_last.shape, dtype=np.uint8)
            for line in lines_parcels:
                p0=line[0]
                p1=line[1]
                # rotate points
                lines_mask = cv2.line(lines_mask,(p0[1],p0[0]),(p1[1],p1[0]),(120,120,120),1)
            split_3d_mask = np.zeros(self.img_last.shape, dtype=np.uint8)
            split_3d_mask[:,:,:] = map.block_mask[:,:,np.newaxis]
            lines_mask = lines_mask & split_3d_mask
            self.img_partitioned = self.img_last.astype(np.uint8) & np.bitwise_not(lines_mask)
            # write partitioning mask
            lines_mask = np.where(lines_mask>0,(255,255,255), (0,0,0))
            if self.partitioning_lines is not None:
                self.partitioning_lines |= lines_mask
            else:
                self.partitioning_lines = lines_mask
        if config.WRITE_UNNECESSARY:
            cv2.imwrite(f'outputs/final_map_{iteration}_{map_id}.bmp',self.img_partitioned)
        self.img_last = self.img_partitioned.copy()
    
    def draw_partitioning_results(self):
        cv2.imwrite(f'outputs/partitioning_mask.bmp',self.partitioning_lines)
        cv2.imwrite(f'outputs/final_map_partitioning.bmp',self.img_partitioned)


    def draw_building(self,building_mask,iteration,map_id,parcel_id,has_building,parcel_type,block_mask):
        color3d_mask = np.zeros(self.img_last.shape, dtype=np.uint8)
        color3d_mask[:,:,:] = block_mask[:,:,np.newaxis]
        if has_building:
            self.total_sum_ff += 1
            color3d_mask = np.where(color3d_mask>0,(100,100,100),(255,255,255))
        elif parcel_type == config.ParcelType.O:
            color3d_mask = np.where(color3d_mask>0,(0,255,255),(255,255,255))
        elif parcel_type == config.ParcelType.A:
            color3d_mask = np.where(color3d_mask>0,(51,255,255),(255,255,255))
        elif parcel_type == config.ParcelType.B:
            color3d_mask = np.where(color3d_mask>0,(102,255,255),(255,255,255))
        elif parcel_type == config.ParcelType.C:
            color3d_mask = np.where(color3d_mask>0,(153,255,255),(255,255,255))
        elif parcel_type == config.ParcelType.U:
            color3d_mask = np.where(color3d_mask>0,(40,0,255),(255,255,255))

        build3d_mask = np.zeros(self.img.shape, dtype=np.uint8)
        build3d_mask[:,:,:] = building_mask[:,:,np.newaxis]
        if self.building_masks is not None:
            self.building_masks &= build3d_mask
            self.img_last &= self.img_partitioned & build3d_mask
            self.img_built &= self.img_partitioned & build3d_mask & color3d_mask
        else:
            self.building_masks = build3d_mask
            self.img_last = self.img_partitioned & build3d_mask
            self.img_built = self.img_partitioned & build3d_mask & color3d_mask
        if config.WRITE_UNNECESSARY:
            cv2.imwrite(f'outputs/final_map_{iteration}_{map_id}_{parcel_id}.bmp',self.img_built)
        
    def draw_building_results(self):
        cv2.imwrite(f'outputs/buildings_mask.bmp',np.bitwise_not(self.building_masks))
        cv2.imwrite(f'outputs/final_map_location_finding.bmp',self.img_built)


    def draw_collision(self):
        trees_mask = cv2.imread('outputs/tree_mask.bmp')
        fixed_facility_mask = cv2.imread('outputs/facility_mask.bmp')
        roads_mask = cv2.imread('outputs/roads_mask.bmp')
        collide_mask = roads_mask
        if os.path.exists('outputs/buildings_mask.bmp'):
            collide_mask |= cv2.imread('outputs/buildings_mask.bmp')
        if os.path.exists('outputs/partitioning_mask.bmp'):
            collide_mask |= cv2.imread('outputs/partitioning_mask.bmp')
        # change here when building mask is av
        collision3dmask = trees_mask | fixed_facility_mask
        collision3dmask = collide_mask & collision3dmask
        img = self.img_last.copy()
        pixels = [100,50,100]*int(len(img[collision3dmask>0])/3)
        img[collision3dmask>0] = pixels

        cv2.imwrite(f'outputs/collision_map.bmp',img)