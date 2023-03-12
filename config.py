import enum
import datetime
# enum classes
class ArchStyles(enum.Enum):
    New_York = 1
    Human_Centered_AI = 2
    Customized = 3
class ParcelType(enum.Enum):
    O = 0 # oversize
    A = 1
    B = 2
    C = 3
    U = 4 # undersize

# main run input args
MAIN_MAP_ADDR = 'inputs/Sample_test120-70_total_map_fringe.bmp'
MAIN_MAP_FILLED_BLOCK_MASK = 'inputs/Sample_test120-70_boundry_mask_fringe.bmp'
MAIN_MAP_FILLED_F_F_MASK = 'inputs/Sample_test120-70_FiFa_mask_fringe.bmp'
PARCELS_COUNT = 12
ARCH = ArchStyles.Human_Centered_AI
ARCH_FIRST_INPUT = (44, 119) #(70, 120) #(Y,X) or None

# Axis finding args
VALID_POINT_DISTANCE = 10
ROAD_SIZE_MAX = 3 # can be: 1,3,5,7,...
ROAD_SIZE_MIN = 1 # can be: 1,3,5,...
ROAD_STEP = 2 # can be multiplication of 2

ACCESS_RATIO = 0.6 # 0.55 for newyork # 0.7 for star

AXIS_MIN_AREA = 1000 # 12000 for newyork # 75000 for star

FACILITY_SAFE_DIST = 2
TREE_SAFE_DIST = 1

# axis finder fitness weights
A_ACCESS_SPLIT_WEIGHT  = 1
A_AREA_SPLIT_WEIGHT = 1
A_FIXED_FACILITIES_WEIGHT = 1
A_CARBON_WEIGHT = 1

# Partitioning args
PROCESSING_CORES = 8

P_GA_LOSS_THOLD = 0.3

TYPE_A_AREA = 700
TYPE_B_AREA = 500
TYPE_C_AREA = 400

TYPE_AREA_STEP = 100

# partiotioning fitness weights
P_FF_WEIGHT = 2
P_AREA_WEIGHT = 2
P_TREE_WEIGHT = 1

# location finding args
BUILDING_ADDR = ['inputs/rectangle_6-15.bmp','inputs/rectangle_6-15.bmp','inputs/rectangle_6-15.bmp','inputs/white.bmp']
BUILDING_BORDER_GAP = 10

# location finding fitness weights
L_DIST_WEIGHT = 1
L_OUT_BORDER_WEIGHT = 2
L_CARBON_WEIGHT = 1
L_ON_ROAD_WEIGHT = 2

def building_sel(p_type:ParcelType):
    if p_type == ParcelType.O: return BUILDING_ADDR[0]
    if p_type == ParcelType.A: return BUILDING_ADDR[0]
    if p_type == ParcelType.B: return BUILDING_ADDR[1]
    if p_type == ParcelType.C: return BUILDING_ADDR[2]
    if p_type == ParcelType.U: return BUILDING_ADDR[3]

# output args and loggers
WRITE_UNNECESSARY = False
WRITE_LOGS = True
def log_axis_finding_settings():
    log(f'====== Axis Finding inputs ======')
    log(f'PARCELS_COUNT : {PARCELS_COUNT}')
    log(f'ARCH : {ARCH}')
    log(f'ARCH_FIRST_INPUT : {ARCH_FIRST_INPUT}')
    log(f'VALID_POINT_DISTANCE : {VALID_POINT_DISTANCE}')
    log(f'ROAD_SIZE_MAX : {ROAD_SIZE_MAX}')
    log(f'ROAD_SIZE_MIN : {ROAD_SIZE_MIN}')
    log(f'ROAD_STEP : {ROAD_STEP}')
    log(f'ACCESS_RATIO : {ACCESS_RATIO}')
    log(f'AXIS_MIN_AREA : {AXIS_MIN_AREA}')
    log(f'FACILITY_SAFE_DIST : {FACILITY_SAFE_DIST}')
    log(f'TREE_SAFE_DIST : {TREE_SAFE_DIST}')
    log(f'A_ACCESS_SPLIT_WEIGHT : {A_ACCESS_SPLIT_WEIGHT}')
    log(f'A_AREA_SPLIT_WEIGHT : {A_AREA_SPLIT_WEIGHT}')
    log(f'A_FIXED_FACILITIES_WEIGHT : {A_FIXED_FACILITIES_WEIGHT}')
    log(f'A_CARBON_WEIGHT : {A_CARBON_WEIGHT}')
def log_partitioning_settings():
    log(f'====== Partitioning inputs ======')
    log(f'PARCELS_COUNT : {PARCELS_COUNT}')
    log(f'PROCESSING_CORES : {PROCESSING_CORES}')
    log(f'P_GA_LOSS_THOLD : {P_GA_LOSS_THOLD}')
    log(f'TYPE_A_AREA : {TYPE_A_AREA}')
    log(f'TYPE_B_AREA : {TYPE_B_AREA}')
    log(f'TYPE_C_AREA : {TYPE_C_AREA}')
    log(f'TYPE_AREA_STEP : {TYPE_AREA_STEP}')
    log(f'P_FF_WEIGHT : {P_FF_WEIGHT}')
    log(f'P_AREA_WEIGHT : {P_AREA_WEIGHT}')
    log(f'P_TREE_WEIGHT : {P_TREE_WEIGHT}')
def log_location_finding_settings():
    log(f'====== Location Finding inputs ======')
    log(f'PARCELS_COUNT : {PARCELS_COUNT}')
    log(f'BUILDING_ADDR : {BUILDING_ADDR}')
    log(f'BUILDING_BORDER_GAP : {BUILDING_BORDER_GAP}')
    log(f'TYPE_A_AREA : {TYPE_A_AREA}')
    log(f'TYPE_B_AREA : {TYPE_B_AREA}')
    log(f'TYPE_C_AREA : {TYPE_C_AREA}')
    log(f'TYPE_AREA_STEP : {TYPE_AREA_STEP}')
    log(f'L_DIST_WEIGHT : {L_DIST_WEIGHT}')
    log(f'L_OUT_BORDER_WEIGHT : {L_OUT_BORDER_WEIGHT}')
    log(f'L_CARBON_WEIGHT : {L_CARBON_WEIGHT}')
    log(f'L_ON_ROAD_WEIGHT : {L_ON_ROAD_WEIGHT}')
def log(msg):
    if WRITE_LOGS:
        with open('./outputs/logs.txt', 'a') as f:
            f.writelines(f'{datetime.datetime.now()} : {str(msg)}\n')
    print(str(msg))