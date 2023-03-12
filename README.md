# LandParcel
This project helps architects input an image of the desired land with blue as a boundary, red as road access, green as trees, and black as a fixed pre-built facility. The project algorithm inputs are defined in ‘configs.py’ file.
## How to Use
Project consists of three parts: 
### 1. Axis Finding
Axis is the main road from a parcel of land, this part decides how many sub parts is needed. (Architecture of devision includes: New York, Customized by starting point, fully customized)
### 2. Partitioning
Partitioning is deviding each sub parts of land into several parts.
### 3. Location Finding
Location Finding finds the best location to build a proper sized house.(houses has three types, sorted by size of them in the input folders)

# Huggin face
https://huggingface.co/spaces/amirDev/LandPartitioning
