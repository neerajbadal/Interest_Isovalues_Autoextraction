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
import sys
import pandas as pd
def normalizeHistPlot(hist_d):
    '''
    normalize the the distribution
    '''
    min_v = np.min(hist_d)
    max_v = np.max(hist_d)
    hist_d = (hist_d - min_v) / float(max_v-min_v)
    return hist_d

def getScalarFrequency(bin_data):
    '''
    returns the scalar frequency histogram of the given data
    '''
    hist_data, bin_edges = np.histogram(pvm_data,bins=np.arange(max_val+1))
    hist_data = np.log(hist_data+1)
    hist_data = normalizeHistPlot(hist_data)
    return hist_data

def getGridEdgeFrequency(bin_data,range_):
    '''
    returns the  grid edge frequency histogram,
                 total gradient edge based frequency,
                 mean gradient edge frequency
    '''
    
    i_z = bin_data.shape[0]
    i_y = bin_data.shape[1]
    i_x = bin_data.shape[2]
    
    hist_ = np.zeros(range_)
    
    hist_gradient = np.zeros(range_)
 
    
    for k in range(0,i_z,1):
        for j in range(0,i_y,1):
            for i in range(0,i_x,1):
                if (i+1) < i_x:
                    modGradient = np.fabs(bin_data[k,j,i]-bin_data[k,j,i+1]) 
                    
                    hist_[bin_data[k,j,i]:bin_data[k,j,i+1]+1] = hist_[bin_data[k,j,i]:bin_data[k,j,i+1]+1] + 1.0
                    
                    hist_gradient[bin_data[k,j,i]:bin_data[k,j,i+1]+1] = hist_gradient[bin_data[k,j,i]:bin_data[k,j,i+1]+1] + (modGradient/3.0)
                    
                    
                if (j+1) < i_y: 
                    hist_[bin_data[k,j,i]:bin_data[k,j+1,i]+1] = hist_[bin_data[k,j,i]:bin_data[k,j+1,i]+1] + 1.0
                    
                    modGradient = np.fabs(bin_data[k,j,i]-bin_data[k,j+1,i])
                    hist_gradient[bin_data[k,j,i]:bin_data[k,j+1,i]+1] = hist_gradient[bin_data[k,j,i]:bin_data[k,j+1,i]+1] + (modGradient/3.0)
                      
                if (k+1) < i_z: 
                    hist_[bin_data[k,j,i]:bin_data[k+1,j,i]+1] = hist_[bin_data[k,j,i]:bin_data[k+1,j,i]+1] + 1.0
                    modGradient = np.fabs(bin_data[k,j,i]-bin_data[k+1,j,i])
                    hist_gradient[bin_data[k,j,i]:bin_data[k+1,j,i]+1] = hist_gradient[bin_data[k,j,i]:bin_data[k+1,j,i]+1] + (modGradient/3.0)
                    
    
    total_gradient = np.copy(hist_gradient)
    mean_gradient = hist_gradient /hist_
    
    total_gradient = np.log(total_gradient+1)
    mean_gradient = np.log(mean_gradient+1) 
    hist_ = np.log(hist_+1)
    
    hist_ = normalizeHistPlot(hist_)
    total_gradient = normalizeHistPlot(total_gradient)
    mean_gradient = normalizeHistPlot(mean_gradient)
    
    return [hist_,total_gradient,mean_gradient]

def getGridEdgeFrequencyV2(bin_data,range_):
    '''
    returns the  grid edge frequency histogram,
                 total gradient edge based frequency,
                 mean gradient edge frequency
    '''
    
    i_z = bin_data.shape[0]
    i_y = bin_data.shape[1]
    i_x = bin_data.shape[2]
    hist_ = np.zeros(range_)
    hist_gradient = np.zeros(range_)
    hist_volume_edge = np.zeros(range_)
    
    '''in x-axis direction'''
    bin_data_curr = bin_data[:,:,:-1]
    bin_data_neighbor = bin_data[:,:,1:]
    x_axis_cross_hist ,x_edges,y_edges= np.histogram2d(bin_data_curr.reshape(-1),bin_data_neighbor.reshape(-1),bins=range_)
    '''in y-axis direction'''
    bin_data_curr = bin_data[:,:-1,:]
    bin_data_neighbor = bin_data[:,1:,:]
    y_axis_cross_hist, x_edges, y_edges = np.histogram2d(bin_data_curr.reshape(-1),bin_data_neighbor.reshape(-1),bins=range_)
    '''in z-axis direction'''
    bin_data_curr = bin_data[:-1,:,:]
    bin_data_neighbor = bin_data[1:,:,:]
    z_axis_cross_hist, x_edges, y_edges = np.histogram2d(bin_data_curr.reshape(-1),bin_data_neighbor.reshape(-1),bins=range_)
    
    for dn_curr in range(z_axis_cross_hist.shape[0]):
        for dn_neighbor in range(z_axis_cross_hist.shape[1]):
            hist_[dn_curr:dn_neighbor+1] = hist_[dn_curr:dn_neighbor+1]+x_axis_cross_hist[dn_curr,dn_neighbor]
            hist_[dn_curr:dn_neighbor+1] = hist_[dn_curr:dn_neighbor+1]+y_axis_cross_hist[dn_curr,dn_neighbor]
            hist_[dn_curr:dn_neighbor+1] = hist_[dn_curr:dn_neighbor+1]+z_axis_cross_hist[dn_curr,dn_neighbor]
            mod_gradient = np.fabs(dn_neighbor-dn_curr)/3
            hist_gradient[dn_curr:dn_neighbor+1] = hist_gradient[dn_curr:dn_neighbor+1]+x_axis_cross_hist[dn_curr,dn_neighbor]*mod_gradient
            hist_gradient[dn_curr:dn_neighbor+1] = hist_gradient[dn_curr:dn_neighbor+1]+y_axis_cross_hist[dn_curr,dn_neighbor]*mod_gradient
            hist_gradient[dn_curr:dn_neighbor+1] = hist_gradient[dn_curr:dn_neighbor+1]+z_axis_cross_hist[dn_curr,dn_neighbor]*mod_gradient
            mod_grad_inverse = 1/(3.0*np.fabs(dn_neighbor-dn_curr))
            
            if dn_curr == dn_neighbor:
                hist_volume_edge[dn_curr] += 1/(3.0) 
            else:
                hist_volume_edge[dn_curr:dn_neighbor+1] = hist_volume_edge[dn_curr:dn_neighbor+1]+z_axis_cross_hist[dn_curr,dn_neighbor]*mod_grad_inverse
    total_gradient = np.copy(hist_gradient)
    mean_gradient = hist_gradient /hist_
    
    total_gradient = np.log(total_gradient+1)
    mean_gradient = np.log(mean_gradient+1) 
    hist_ = np.log(hist_+1)
    
    hist_ = normalizeHistPlot(hist_)
    total_gradient = normalizeHistPlot(total_gradient)
    mean_gradient = normalizeHistPlot(mean_gradient)
    
    hist_volume_edge = np.log(hist_volume_edge+1)
    hist_volume_edge = normalizeHistPlot(hist_volume_edge)
    
    return [hist_,total_gradient,mean_gradient,hist_volume_edge]


def getGridCubeFrequency(bin_data,range_):
    '''
    returns the computed grid edge frequency histogram 
    '''
    
    i_z = bin_data.shape[0]
    i_y = bin_data.shape[1]
    i_x = bin_data.shape[2]
    
    hist_ = np.zeros(range_)
    
    for k in range(0,i_z,1):
        for j in range(0,i_y,1):
            
            for i in range(0,i_x,1):
                minScalarValue = range_ + 2
                maxScalarValue = -9999
                cubeScalarValues = list()
                if (i+1) < i_x and (j+1) < i_y and (k+1) < i_z: 
                    cubeScalarValues.append(bin_data[k,j,i])
                    cubeScalarValues.append(bin_data[k,j,i+1])
                    cubeScalarValues.append(bin_data[k+1,j,i+1])
                    cubeScalarValues.append(bin_data[k,j+1,i+1])
                    cubeScalarValues.append(bin_data[k+1,j+1,i+1])
                    cubeScalarValues.append(bin_data[k+1,j+1,i])
                    cubeScalarValues.append(bin_data[k,j+1,i])
                    cubeScalarValues.append(bin_data[k+1,j,i])
                
                if len(cubeScalarValues) > 0:
                    for val in cubeScalarValues:
                        if val < minScalarValue:
                            minScalarValue = val    
                        if val > maxScalarValue:
                            maxScalarValue = val
                    hist_[minScalarValue:maxScalarValue+1] = hist_[minScalarValue:maxScalarValue+1] + 1
                cubeScalarValues.clear()
    
    hist_ = np.log(hist_)
    hist_ = normalizeHistPlot(hist_)
    return hist_

def getGridCubeFrequencyV2(bin_data,range_):
    '''
    returns the computed grid edge frequency histogram 
    '''
    
    
    from skimage.util import view_as_windows
    voxels_ = view_as_windows(bin_data,(2,2,2), step=1)
    
    voxels_ = voxels_.reshape(voxels_.shape[0],
                              voxels_.shape[1],
                              voxels_.shape[2],
                              (2*2*2)
        )
    voxels_ = voxels_.reshape(voxels_.shape[0]*
                              voxels_.shape[1]*voxels_.shape[2],voxels_.shape[3]
        )
    
#     mimMaxList = [[np.min(vox_d),np.max(vox_d)] for vox_d in voxels_]
    minList = np.min(voxels_,1)
    maxList = np.max(voxels_,1)
    h_k = np.zeros((range_,range_))
    h_k_cube = np.zeros((range_,range_))
    hist_ = np.zeros(range_)
    hist_vol_cube = np.zeros(range_)
    
    minList = np.array(minList)
    maxList = np.array(maxList)    
    for i_ in range(0,len(minList)):
        if minList[i_]==maxList[i_]:
            h_k_cube[minList[i_],maxList[i_]] += 1.0
        else:
            h_k_cube[minList[i_],maxList[i_]] += (1.0/(maxList[i_]-minList[i_]))
        
        h_k[minList[i_],maxList[i_]] = h_k[minList[i_],maxList[i_]] + 1
#     h_k[minList[:],maxList[:]] = h_k[minList[:],maxList[:]] + 1
#     minList = np.array(minList)
#     maxList = np.array(maxList)
#     x_axis_cross_hist ,x_edges,y_edges= np.histogram2d(minList,maxList,bins=(range_,range_))    
    
    for i in range(0,range_):
        for j in range(0,range_):
            hist_[i:j+1] += h_k[i,j]
            hist_vol_cube[i:j+1] += h_k_cube[i,j]
                
                
#     h_k = hist_
    hist_ = np.log(hist_+1)
    hist_ = normalizeHistPlot(hist_)
    
    hist_vol_cube = np.log(hist_vol_cube+1)
    hist_vol_cube = normalizeHistPlot(hist_vol_cube)
    
    return [hist_,hist_vol_cube]


def getTriangleCount_Par(bin_data,level_):

    try:
        
        verts, faces, normals, values = measure.marching_cubes_lewiner(bin_data, level=level_, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=1, allow_degenerate=True, use_classic=False)
        tri_count = len(faces)
        iso_area = measure.mesh_surface_area(verts,faces)
    except:
        return [0,0]
    return [tri_count,iso_area]
    
    

def getTriangle_IsosurfaceCount(bin_data,range_):
    '''
    returns the computed grid edge frequency histogram 
    '''
    triangle_count = []
    isosurface_area = []
    
    '''Triangle count and isosurface area estimation using marching cubes'''


#     for i in range(0,range_-1):
#         print(i)
#         verts, faces, normals, values = measure.marching_cubes_lewiner(bin_data, level=i, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=1, allow_degenerate=True, use_classic=False)
#         triangle_count.append(len(faces))
#         isosurface_area.append(measure.mesh_surface_area(verts,faces))
    
    pool = multiprocessing.Pool(3)
    results = pool.map(partial(getTriangleCount_Par, bin_data), np.arange(0,range_-1))
    pool.close()
    pool.join()
#     verts, faces, normals, values = zip(*(measure.marching_cubes_lewiner(bin_data, level=i,
#                             spacing=(1.0, 1.0, 1.0),
#                              gradient_direction='descent',
#                               step_size=1,
#                                allow_degenerate=True,
#                                 use_classic=False) for i in range(0,range_-1)))
    
    
    results = np.array(results)
   
    triangle_count = np.array(results[:,0])
    isosurface_area = np.array(results[:,1])
    triangle_count = np.log(triangle_count+1)
    triangle_count = normalizeHistPlot(triangle_count)
        
    isosurface_area = np.log(isosurface_area+1)
    isosurface_area = normalizeHistPlot(isosurface_area)
        
        
    return [triangle_count,isosurface_area]



def computeIntervalVolume(bin_data,range_):
    '''Interval volume using grid edges'''
    grid_edge_count=[]
    '''isovalue bins starting from 0 to 256-2, 0 bin implies [0,1],1 bin implies [1,2]'''
    for index in range(0,np.amax(bin_data)):
        grid_edge_count.append(0)
    
    
    #Edges lie in three orientations along x,y,z axes
    for axis in range(0,3):
        if axis==0:
            x=1
            y=0
            z=0
        if axis==1:
            x=0
            y=1
            z=0
        if axis==2:
            x=0
            y=0
            z=1
        for i in range(0,len(bin_data)-(1+x)):
            for j in range(0,len(bin_data)-(1+y)):
                for k in range(0,len(bin_data[0][0])-(1+z)):
                    #Edge scalars below store the scalars at the end points scalar values
                    dummy1=bin_data[i][j][k]
                    dummy2=bin_data[i+x][j+y][k+z]
                    '''
                    Assumptions:
                    1.If an edge has smaller value>sigma0 and largervalue<sigma1, it is counted as 1/3
                    2.If an interval 
                    rearranging scalar values to get min in scalar1 and max in scalar2
                    '''
                    edge_scalar1=min(dummy1,dummy2)
                    edge_scalar2=max(dummy1,dummy2)
                    if edge_scalar1==edge_scalar2:
                        '''
                        need to change below to one of the ranges
                        '''
                        grid_edge_count[edge_scalar1-1]+=1/3
#                         grid_edge_count[edge_scalar1]+=1/3
                    else:
                        for scalar in range(edge_scalar1,edge_scalar2):
                            grid_edge_count[scalar]+=(1/3)*(1/(edge_scalar2-edge_scalar1))


    grid_edge_count = np.array(grid_edge_count)
    grid_edge_count = np.log(grid_edge_count+1)
    grid_edge_count = normalizeHistPlot(grid_edge_count)

    return grid_edge_count


if __name__ == '__main__':
    
    
    if len (sys.argv) != 3 :
        print("Usage: python main.py <absoluteInputDataDirectory> <inputDataListFileName>")
        sys.exit(1)
    baseDirectoryPath = sys.argv[1]
    inputFile = sys.argv[2]
    
    
#     baseDirectoryPath = "D:/Mtech/FY/Graphics/GraphicsProjectData/"
    
    fileName = pd.read_csv(inputFile,delimiter='\n',header=None)
    
    fileName = fileName[0].values.tolist()

#     fileName = [
#     "tooth_103_94_161_uint8.raw",   
#     "stent_512_512_174_uint16.raw",
#     "mrt-angio_416_512_112_uint16.raw",        
#     "Orange_256_256_64_uint8.raw",
#     "Tomato_256_256_64_uint8.raw"
#     "Backpack_512_512_373_uint8.raw",
#     "Angio_384_512_80_uint8.raw",
#     "CT-Knee_379_229_305_uint8.raw",
#     "DTI-MD_128_128_58_uint8.raw",
#     "MRI-Head_256_256_256_uint8.raw",
#     "Tumor-Breast_448_448_208_uint8.raw",
#     "Tumor-DCIS_448_448_160_uint8.raw",
#     "visible-male_128_256_256_uint8.raw",
#     
#     "Knee_512_512_87_uint16.raw",
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
    
  
    
    for fileName_ in fileName:
        
       
        nameSplits = fileName_.split('_')
        print('----------File Info---------')
        print(nameSplits)
        classNameRep = nameSplits[0]#"Kidney"
        pvmFile = baseDirectoryPath +"/"+fileName_
        
        dataType = nameSplits[4].split('.')[0]
        if dataType == "uint8":
            pvm_data = np.fromfile(pvmFile,dtype=np.uint8)
        elif dataType == "uint16":
            pvm_data = np.fromfile(pvmFile,dtype=np.uint16)
       
        max_val = np.max(pvm_data)
        
        
        max_val = np.int32((2**np.ceil(np.log2(max_val)))-1)
        
        grid_x = int(nameSplits[1])#256
        grid_y = int(nameSplits[2])#256
        grid_z = int(nameSplits[3])#256
        
        time_list = []
        
        starttime = time.time()
        scalar_freq = getScalarFrequency(pvm_data)
        endTime = time.time()
        print("scalar frequency generation time : ",endTime-starttime)
        time_list.append(endTime-starttime)
        pvm_data = pvm_data.reshape(grid_z,grid_y,grid_x)
        range_ = max_val + 1
        
       
        
        pvm_data = pvm_data.astype(np.int32)
        
        starttime = time.time()
        freq_plots = getGridEdgeFrequencyV2(pvm_data, range_)
        endTime = time.time()
        print("Edge and Gradient frequency generation time : ",endTime-starttime)
        time_list.append(endTime-starttime)
        edge_freq = freq_plots[0]
        total_gradient_freq = freq_plots[1]
        mean_gradient_freq = freq_plots[2]
        hist_vol_edge = freq_plots[3]
        
        
        starttime = time.time()
        cube_freq = getGridCubeFrequencyV2(pvm_data, range_)
        endTime = time.time()
        print("Cube frequency generation time : ",endTime-starttime)
    #     plt.plot(cube_freq[1])
    #     plt.show()
         
        time_list.append(endTime-starttime)
        
        
        starttime = time.time()
        triangle_isoArea = getTriangle_IsosurfaceCount(pvm_data, range_)
        endTime = time.time()
        print("Triangle Count and IsoArea frequency generation time : ",endTime-starttime)
        time_list.append(endTime-starttime)
         
        triangleCount = triangle_isoArea[0]
        isoAreaCount = triangle_isoArea[1]
        
        fileRepresentative = fileName_.split('.')[0]
         
        np.savetxt(fileRepresentative+"_scalarFreq_.dat",scalar_freq)
        np.savetxt(fileRepresentative+"_edgeFreq_.dat",edge_freq)
        np.savetxt(fileRepresentative+"_cubeFreq_.dat",cube_freq)
        np.savetxt(fileRepresentative+"_totalGradientFreq_.dat",total_gradient_freq)
        np.savetxt(fileRepresentative+"_meanGradientFreq_.dat",mean_gradient_freq)
        np.savetxt(fileRepresentative+"_triangleFreq_.dat",triangleCount)
        np.savetxt(fileRepresentative+"_isoAreaFreq_.dat",isoAreaCount)
        np.savetxt(fileRepresentative+"_hist_vol_edgeFreq_.dat",hist_vol_edge)
        np.savetxt(fileRepresentative+"_hist_vol_cubeFreq_.dat",cube_freq[1])
        
        np.savetxt(fileRepresentative+"_time_info_tria.dat",time_list)
        
        
        
        
        
        plt.figure(figsize=(9,6))
        plt.plot(scalar_freq,ls='-.',label="scalar frequency")
        plt.plot(edge_freq,ls=':',label="iso-surface area (edge based)")
        plt.plot(cube_freq[0],ls='--',label="iso-surface area (cube based)")
        plt.plot(total_gradient_freq,ls='--',label="Total Gradient (edge based)")
        plt.plot(mean_gradient_freq,ls='-',label="Mean Gradient (edge based)")
        plt.plot(triangleCount,ls='-',label="Triangle Count")
        plt.plot(isoAreaCount,ls='-',label="Isosurface Area Count")
        plt.plot(hist_vol_edge,ls='--',label="iso-surface volume (edge based)")
        plt.plot(cube_freq[1],ls='--',label="iso-surface volume (cube based)")
         
        plt.xlabel("scalar values")
        plt.ylabel("frequency")
        plt.title(classNameRep+" Dataset Frequency Comparison")
        plt.legend()
        plt.grid()
         
#         plt.legend(bbox_to_anchor=(0,-0.8,1,0.3), loc="lower left",mode="expand",borderaxespad=0,title="WL = Wickets Lost",ncol=4)
        plt.tight_layout()
        plt.savefig(fileRepresentative+"_comp_plot.png")
    
        plt.figure(figsize=(9,6))
        plt.plot(scalar_freq,ls='-.',label="scalar frequency")
        plt.plot(edge_freq,ls=':',label="iso-surface area (edge based)")
        plt.plot(cube_freq[0],ls='--',label="iso-surface area (cube based)")
        plt.plot(triangleCount,ls='-',label="Triangle Count")
        plt.plot(isoAreaCount,ls='-',label="Isosurface Area Count")
        
        plt.xlabel("scalar values")
        plt.ylabel("frequency")
        plt.title(classNameRep+" Dataset Frequency Comparison Triangle Count Iso-Surface Area Based")
        plt.legend()
        plt.grid()
        
#         plt.legend(bbox_to_anchor=(0,-0.8,1,0.3), loc="lower left",mode="expand",borderaxespad=0,title="WL = Wickets Lost",ncol=4)
        plt.tight_layout()
        plt.savefig(fileRepresentative+"_comp_plot_isa_tri.png")
    
    
        plt.figure(figsize=(9,6))
        plt.plot(total_gradient_freq,ls='--',label="Total Gradient (edge based)")
        plt.plot(mean_gradient_freq,ls='-',label="Mean Gradient (edge based)")
         
        plt.xlabel("scalar values")
        plt.ylabel("frequency")
        plt.title(classNameRep+" Dataset Frequency Comparison Gradient Based")
        plt.legend()
        plt.grid()
         
#         plt.legend(bbox_to_anchor=(0,-0.8,1,0.3), loc="lower left",mode="expand",borderaxespad=0,title="WL = Wickets Lost",ncol=4)
        plt.tight_layout()
        plt.savefig(fileRepresentative+"_comp_plot_grad.png")
#         plt.show()
    
    
    
        plt.figure(figsize=(9,6))
         
        plt.plot(hist_vol_edge,ls='--',label="iso-surface volume (edge based)")
        plt.plot(cube_freq[1],ls='--',label="iso-surface volume (cube based)")
         
        plt.xlabel("scalar values")
        plt.ylabel("frequency")
        plt.title(classNameRep+" Dataset Frequency Comparison Volume Based")
        plt.legend()
        plt.grid()
         
#         plt.legend(bbox_to_anchor=(0,-0.8,1,0.3), loc="lower left",mode="expand",borderaxespad=0,title="WL = Wickets Lost",ncol=4)
        plt.tight_layout()
        plt.savefig(fileRepresentative+"_comp_plot_vol.png")
        
        print(fileName_+"  analysis plots generated")
        print('---------------------------')
        
    
    
    
    
#     np.savetxt("scalarFreq_kidney.dat",hist_tooth)
    