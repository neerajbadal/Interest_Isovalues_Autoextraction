'''
Created on 06-Dec-2019

@author: Neeraj Badal
'''
'''
Created on 16-Nov-2019

@author: Neeraj Badal
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage.draw import ellipsoid
import multiprocessing
from functools import partial
from _ast import Try
import pandas as pd
import sys

if __name__ == '__main__':
    
    
    if len (sys.argv) != 3 :
        print("Usage: python main.py <absoluteInputDataDirectory> <inputDataListFileName>")
        sys.exit(1)
    baseDirectoryPath = sys.argv[1]
    inputFile = sys.argv[2]
    
    
#     baseDirectoryPath = "D:/Mtech/FY/Graphics/GraphicsProjectData/"
    
    fileName = pd.read_csv(inputFile,delimiter='\n',header=None)
    
    fileName = fileName[0].values.tolist()
    
#     baseDirectoryPath = "D:/workspace/GraphicsProject/PVMReader/Results/"
    
#     fileName = [
#     "tooth_103_94_161_uint8.raw",   
#     "stent_512_512_174_uint16.raw",
#     "mrt-angio_416_512_112_uint16.raw",        
#     "Orange_256_256_64_uint8.raw",
#     "Tomato_256_256_64_uint8.raw"
#     "Backpack_512_512_373_uint8.raw",
# # 
#     "Angio_384_512_80_uint8.raw",
#     "CT-Knee_379_229_305_uint8.raw",
# # 
#     "DTI-MD_128_128_58_uint8.raw",
# # 
#     "MRI-Head_256_256_256_uint8.raw",
#     "Tumor-Breast_448_448_208_uint8.raw",
#     "Tumor-DCIS_448_448_160_uint8.raw",
#     "visible-male_128_256_256_uint8.raw",
#     
# #     "Knee_512_512_87_uint16.raw",
#     "aneurism_256_256_256_uint8.html",
#     "bonsai_256_256_256_uint8.html",
#     "foot_256_256_256_uint8.html",
#     "engine_256_256_128_uint8.raw",
#     "fuel_64_64_64_uint8.raw",
#     "kidney_384_384_96_uint8.raw",
#     "marschner-lobb_41_41_41_uint8.raw",
#     "skull_256_256_256_uint8.raw",
#     "statue-leg_341_341_93_uint8.raw"
#     ]
    
    timeVals = []
    for fileName_ in fileName:
        
        nameSplits = fileName_.split('_')
        print(nameSplits)
        classNameRep = nameSplits[0]#"Kidney"
        timeFileName = fileName_.split('.')[0] + "_time_info.dat"
        timeFileTriangle = fileName_.split('.')[0] + "_time_info_tria.dat"
        
        timeFileName = baseDirectoryPath +"/"+timeFileName
        
        timeFileTriangle = baseDirectoryPath +"/"+timeFileTriangle
        
        timeFileDat = pd.read_csv(timeFileName,delimiter='\n',header=None)
        
        timeFileTria = pd.read_csv(timeFileTriangle,delimiter='\n',header=None)
        
        df = timeFileDat.append(timeFileTria)
        
        val = df[0].values
        
        timeVals.append([classNameRep,val[0],val[1],val[2],val[3]])
    
    
    
    timeVals = np.array(timeVals)
    print(timeVals)
    
    plt.title("Execution Time after Optimization")
    plt.xlabel("Data Set")
    plt.ylabel("Execution time in sec.")
    plt.plot(timeVals[:,1].astype(np.float),marker='o',label="scalar frequency")
    plt.plot(timeVals[:,2].astype(np.float),marker='o',label="Edge Based")
    plt.plot(timeVals[:,3].astype(np.float),marker='o',label="Cube Based")
    plt.plot(timeVals[:,4].astype(np.float),marker='o',label="Triangle Count")
    plt.xticks(np.arange(0,len(timeVals)),timeVals[:,0],rotation="vertical")
    plt.legend()
    plt.grid()
    plt.show()
    
