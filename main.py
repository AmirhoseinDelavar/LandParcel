from axis_finder import run_axis_finder
from partitioning import run_partitioning
from location_finder import run_location_finding
import config

if __name__ == '__main__':
    # Axis Finding Step
    total_execution_time = 0
    total_execution_time = run_axis_finder(total_execution_time)
    # Partitioning Step
    total_execution_time = run_partitioning(total_execution_time)
    # Location Finding Step
    total_execution_time = run_location_finding(total_execution_time)
    # Report Generation Step
    config.log("Total Execution Time: %s seconds or %s minutes" % (total_execution_time, total_execution_time/60))