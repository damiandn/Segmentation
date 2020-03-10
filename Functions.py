
from __future__ import division
from aicsimageio import AICSImage, omeTifWriter
from aicssegmentation.core.visual import seg_fluo_side_by_side,  single_fluorescent_view, segmentation_quick_view
from aicssegmentation.core.vessel import filament_2d_wrapper, filament_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from aicssegmentation.core.utils import get_middle_frame, hole_filling, get_3dseed_from_mid_frame, get_3dseed_from_all_frames
from IPython.display import clear_output, display
from itkwidgets import view
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
#import napari
import numpy as np
from numpy import linalg
import os
import pandas as pd
from pathlib import Path
import pickle
from PIL import Image
from random import random, randrange
from scipy import ndimage as ndi
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.transform import Rotation as R
from skimage import io, measure, data
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, watershed, dilation, ball
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, ball, disk, erosion, dilation   # function for post-processing (size filter)
from statistics import mode

plt.rcParams["figure.figsize"] = [16, 12]
whole_centroid = (77, 531, 132)
shape = (154, 1024, 256)
cmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'BlueViolet', 'DarkOrange', 'DarkSlateGray', 'b']

cellColorDict = {}

for i in range (0, 155):
    cellColorDict[i] = (random(), random(), random())

class DDN:

    class Utils:

        """Helper functions for various  tasks
        """
        def TestFunctionsLoaded():
            """test to see if the functions loaded correctly
            """
            print('Functions Loaded!')

        def GetCentroid(myDict, track, timepoint):
            """Given a cell and a timepoint, will return a centroid position from the dict
            """
            centroid = myDict[track][timepoint]['Centroid']
            return centroid

        def GetNumberOfTracks(dict):
            """Returns the total number of tracks in a dict"""
            counter = 0
            counter = 0
            for track in dict.keys():
                 counter = counter + 1
            return counter

        def LoadDictByFileName(filename):
            """Loads a specific Dictionary by filename"""
            with open (filename, 'rb') as fp:
                Dict = pickle.load(fp)
            return Dict

        def AddCellToImage(dict, cells, timepoint, img=None, value=255):
            """adds a cell/timepoint to an image, from a dictionary
                if an img is supplied, will add to that image, otherwise, will make a new image
                dict: Dictionary containing cell data
                cells: single cell (int) or list of cells to add
                timepoint: timepoint to add cells from
                img (optional): image to add cells to. If none supplied, will return a new image
                value (optional): value to fill cell at (if none, will default to 255)
            """
            if img is None:
                img = np.ma.array(np.zeros(shape))

            if type(cells) == int:
                if timepoint in dict[cells].keys():
                        coord = dict[cells][timepoint]['Coordinates']
                        for inp in coord:
                            img[inp[0],inp[1],inp[2]]=value
                return img

            if type(cells) == list:
                for cell in cells:
                    if timepoint in dict[cell].keys():
                        coord = dict[cell][timepoint]['Coordinates']
                        for inp in coord:
                            img[inp[0],inp[1],inp[2]]=value
                return img

        def AddNucleusToImage(dict, cells, timepoint, img=None, value=255):
            """adds a nucleus/timepoint to an image, from a dictionary
            if an img is supplied, will add to that image, otherwise, will make a new image
            dict: Dictionary containing nucleus data
            cells: single nucleus (int) or list of nuclei to add
            timepoint: timepoint to add nuclei from
            img (optional): image to add nuclei to. If none supplied, will return a new image
            value (optional): value to fill nucleus at (if none, will default to 255)
            """
            if img is None:
                img = np.ma.array(np.zeros(shape))

            if type(cells) == int:
                if timepoint in dict[cells].keys():
                        coord = dict[cells][timepoint]['Nucleus']['Coordinates']
                        for inp in coord:
                            img[inp[0],inp[1],inp[2]]=value
                return img

            if type(cells) == list:
                for cell in cells:
                    if timepoint in dict[cell].keys():
                        coord = dict[cell][timepoint]['Nucleus']['Coordinates']
                        for inp in coord:
                            img[inp[0],inp[1],inp[2]]=value
                return img

        def SaveDictToFileName(filename, dict):
            """saves a dictionary to a filename"""
            with open (filename, 'wb') as fp:
                pickle.dump(dict, fp)

        def OpenTiff(filename):
            """Helper function to open a Tiff file"""
            im = io.imread(filename)
            imarray = np.array(im)
            return imarray

        def GetSurroundList(myDict, cellTrack, timepoint, range = 25):
            """Gets the cells within a given range of a cell at a timepoint
            myDict: Dictionary containing cell data
            CellTrack: cell to use as center point
            timepoint: timepoint to return
            range: Range within which a cell must be within to be included (default = 25)
            returns: a list of cell track IDs within range of the selected cell
            """
            surround_list = []
            centroid = myDict[cellTrack][timepoint]['Centroid']
            for track in myDict.keys():
                if timepoint in myDict[track].keys():
                    dist = DDN.Utils.calculateDistance(np.array(centroid), np.array(myDict[track][timepoint]['Centroid']   ))
                    if  dist < range:
                        surround_list.append(track)
            return surround_list

        def LoadCurrentDicts():
            """Load the current dictionary (stored in 'current_active_dict' in the 'output' folder.
            Returns:
            cell data dictionary
            cell mesh dictionary (high res)
            nuclear mesh dictionary (high res)
            cell mesh dictionary (low res)
            """
            with open ('output\\DictWithShapeFactor', 'rb') as fp:
                cellData = pickle.load(fp)
            with open ('output\\current_mesh_dict', 'rb') as fp:
                cellMeshData = pickle.load(fp)
            with open ('output\\nuc_mesh_dict', 'rb') as fp:
                nucMeshData = pickle.load(fp)
            with open ('output\\low_poly_cell_mesh', 'rb') as fp:
                lowPolyCellMesh = pickle.load(fp)
            print('cell and mesh dictionaries loaded!')
            return cellData, cellMeshData, nucMeshData, lowPolyCellMesh

        def calculateDistance(p1, p2):
            """return the distance between two np array centroids, p1 and p2
            """
            squared_dist = np.sum((p1-p2)**2, axis=0)
            dist = np.sqrt(squared_dist)
            return dist

        def makeImageFromCoordinates(coordinates):
            """ Takes input coordinates from a dict or prop, and an image shape, and returns a numpy array containing the image
                coordinates: A list of Z,Y,Z coordinates
                shape: a tuple of the Z,Y,Z dimensions of the image'''
            """
            img = np.ma.array(np.zeros(shape))
            for inp in coordinates:
                img[inp[0],inp[1],inp[2]]=1
            return img


        def makeImageFromDict(dict, cells, timepoint):
            """ Takes input cell array and timepoint, returns npArray image of cells in cell array and timepoint.
                coordinates: A list of Z,Y,Z coordinates
                shape: a tuple of the Z,Y,Z dimensions of the image'''
            """
            img = np.ma.array(np.zeros(shape))

            for cell in cells:
                if timepoint in dict[cell].keys():
                    coord = dict[cell][timepoint]['Coordinates']
                    for inp in coord:
                        img[inp[0],inp[1],inp[2]]=1
            return img

        def makeWholeTimepointImage(dict, timepoint):
            """ Makes a labelled image of all cells from a dict
            """
            img = np.ma.array(np.zeros(shape))
            for cell in dict.keys():
                if timepoint in dict[cell].keys():
                    coord = dict[cell][timepoint]['Coordinates']
                    for inp in coord:
                        img[inp[0],inp[1],inp[2]]=cell
            return img
            

        def saveImageFile(data, savefile):
            """
            save a numpy matrix as a tiff image
            """
            out=data.astype(np.uint8)
            writer = omeTifWriter.OmeTifWriter(savefile)
            writer.save(out)

        def saveImageFileBinary(data, savefile):
            """
            save a numpy matrix as a tiff image in binary format (all non-zero pixels = 255)
            """
            data = data >0
            out=data.astype(np.uint8)
            out[out>0]=255
            writer = omeTifWriter.OmeTifWriter(savefile)
            writer.save(out)

        def FindAverageAndMinimumDistancesWithinFrame(myDict, frame):
            """
            returns the average and minimum distance between all object centroids in a frame.
            """
            
            
            min_dist_array = []
            min_dist_global= 99999

            for cell1 in dict[frame].keys():
                min_dist, label, centroid = findClosestCentroid(cell1, frame, frame)
                min_dist_array.append(min_dist)

            print('average minuimum distance = ' + str(np.average(min_dist_array)))
            print('absolute minuimum distance = ' + str(np.amin(min_dist_array)))

            return np.average(min_dist_array), np.amin(min_dist_array)

        def ReturnVectorDataFromDict(myDict, track, timepoint):
            """
            Helper function to return all of the vector data from a dict
            """
            x = myDict[track][timepoint]['Nucleus']['Ellipsoid_X']
            y = myDict[track][timepoint]['Nucleus']['Ellipsoid_Y']
            z = myDict[track][timepoint]['Nucleus']['Ellipsoid_Z']
            x_values = myDict[track][timepoint]['Nucleus']['Ellipsoid_Long_Axis_X']
            y_values = myDict[track][timepoint]['Nucleus']['Ellipsoid_Long_Axis_Y']
            z_values = myDict[track][timepoint]['Nucleus']['Ellipsoid_Long_Axis_Z']

            return x, y, z, x_values, y_values, z_values


        def GetBoundaryVoxelNumber(dict, timepoint):
            labs = DDN.Utils.makeWholeTimepointImage(dict, timepoint)
            for track in dict.keys():
                if timepoint in dict[track]:
                    coords = dict[track][timepoint]['Coordinates']
                    counter = 0      
                    for item in range(0, len(coords)):
                        coord = coords[item]  
                        #clear_output(wait=True)
                        if DDN.Utils.isPixelEdgeWithOtherLab(labs, coord[0], coord[1], coord[2]):
                            #print('found edge!')
                            counter = counter + 1
                            #surface_voxels.append(item)        
                            #labs[coord[0], coord[1], coord[2]] = 0
                    #tracked_cell_dict[track][timepoint]['Surface'] = counter   
        #clear_output(wait=True)
                    clear_output(wait=True)
                    print('cell number ' + str(track) + ' has a surface area of ' + str(counter) + ' at timepoint ' + str(timepoint))
 
 
        def isPixelEdge(labs, z, y, x):
            label = labs[z,y,x] 
            top=labs[z-1, y, x]
            bottom=labs[z+1, y, x]
            up=labs[z, y-1, x]
            down=labs[z, y+1,x]
            left=labs[z,y,x-1]
            right=labs[z,y,x+1]  
            if top==label:
                if bottom==label:
                    if up==label:
                        if down==label:
                            if left==label:
                                if right==label:
                                    return False
            return True

            def MakeFullContactImage(myDict, timepoint):
                img = np.ma.array(np.zeros(shape))
                for cell in myDict.keys():
                    for contact in myDict[cell][timepoint].keys():
                        for coord in myDict[cell][timepoint][contact]:
                            if contact == 0:
                                img[coord[0],coord[1],coord[2]]=255
                            if contact != 0:
                                img[coord[0],coord[1],coord[2]]=contact
                return img

            def MakeCellContactImage(myDict, timepoint, cells = []):
                img = np.ma.array(np.zeros(shape))
                if len(cells) != 0:
                    for cell in cells:
                        if timepoint in myDict[cell].keys():
                            for contact in myDict[cell][timepoint].keys():
                                for coord in myDict[cell][timepoint][contact]:
                                    if contact == 0:
                                        img[coord[0],coord[1],coord[2]]=255
                                    if contact != 0:
                                        img[coord[0],coord[1],coord[2]]=contact
                return img

        def GetAllCellContacts(myDict, cell):
            contact_list = []
            if cell in myDict.keys():
                for timepoint in myDict[cell].keys():
                    for contact in myDict[cell][timepoint]:
                        if contact not in contact_list:
                            contact_list.append(contact)
                            
            return contact_list, len(contact_list)


    class Images:
        """Functions to make images from meshes using matplotlib
        """

        def GetMesh(myDict, track, timepoint, linewidth = 0.4, alpha = 0.3, edgecolor = 'k', facecolor = 'w'):
            verts = myDict[track][timepoint]['verts']
            faces = myDict[track][timepoint]['faces']
            mesh = Poly3DCollection(verts[faces], linewidths=linewidth, alpha=alpha)
            mesh.set_edgecolor(edgecolor)
            mesh.set_facecolor(facecolor)
            return mesh

        def SaveMeshImageClose(cell_nuc_dict, mesh_dict_cell, mesh_dict_nuc, CellToTrack, timepoint, savefile, cell=True, nuclei=False, surround_cell=False,
        surround_nuclei=False, vector=False, surround_vectors=False, outline=False, TrackCell=False, centroid=None, max_range = 35, colors = ('DarkOrange', 'BlueViolet', 'k', 'r'),
        alpha_list = (0.3, 1.0, 0.03, 0.03), surround_list = [], showFig = False):

            if TrackCell == True:
                centroid = cell_nuc_dict[CellToTrack][timepoint]['Centroid']

            if centroid == None:
                centroid = cell_nuc_dict[CellToTrack][timepoint]['Centroid']

            if len(surround_list) == 0:
                surround_list = DDN.Utils.GetSurroundList(cell_nuc_dict, CellToTrack, timepoint)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            if cell==True:
                mesh = DDN.Images.GetMesh(mesh_dict_cell, CellToTrack, timepoint, 0.4, alpha_list[0], colors[0], 'w')
                ax.add_collection3d(mesh)

            if nuclei==True:
                mesh = DDN.Images.GetMesh(mesh_dict_nuc, CellToTrack, timepoint, 0.4, alpha_list[0], colors[1], 'w')
                ax.add_collection3d(mesh)

            if surround_cell==True:
                for track in surround_list:
                    mesh = DDN.Images.GetMesh(mesh_dict_cell, track, timepoint, 0.4, alpha_list[2], 'k', 'w')
                    ax.add_collection3d(mesh)

            if surround_nuclei==True:
                 for track in surround_list:
                    mesh = DDN.Images.GetMesh(mesh_dict_nuc, track, timepoint, 0.4, alpha_list[2], 'k', 'w')
                    ax.add_collection3d(mesh)

            if vector==True:
                x, y, z, x_values, y_values, z_values = DDN.Utils.ReturnVectorDataFromDict(cell_nuc_dict, CellToTrack, timepoint)
                ax.plot(x_values, y_values, z_values, color='blue')
                ax.plot_wireframe(x, y, z,  rstride=3, cstride=3, color='g', alpha=0.4)

            if surround_vectors==True:
                for track in surround_list:
                    x, y, z, x_values, y_values, z_values = DDN.Utils.ReturnVectorDataFromDict(cell_nuc_dict, track, timepoint)
                    ax.plot(x_values, y_values, z_values, color='red')
                    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.1)

            if outline==True:
                verts = mesh_dict[1][timepoint]['Verts']
                faces = mesh_dict[1][timepoint]['Faces']
                mesh = Poly3DCollection(verts[faces], linewidths=0.3, alpha=alpha_list[3])
                mesh.set_edgecolor('k')
                mesh.set_facecolor('w')
                ax.add_collection3d(mesh)

            ax.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
            ax.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
            ax.set_zlim(centroid[2] - max_range, centroid[2] + max_range)

            ax.autoscale_view()

            plt.tight_layout()

            savefile = 'output\\' + savefile + '_cell' + str(CellToTrack) + '_tp' + str(timepoint) + ".png"

            plt.savefig(savefile,bbox_inches='tight')

            if showFig == False:
                plt.close(fig)

        
        
        
        def makeSurfaceContactImage(trackedDict, meshDict, cell, timepoint):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
                
            centroid = trackedDict[cell][timepoint]['Centroid']
                
            max_range = 30
                
            ax.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
            ax.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
            ax.set_zlim(centroid[2] - max_range, centroid[2] + max_range)
                
            if cell in meshDict.keys() and timepoint in meshDict[cell].keys():
                for contact in meshDict[cell][timepoint]:
                    verts = meshDict[cell][timepoint][contact]['verts']
                    faces = meshDict[cell][timepoint][contact]['faces']
                    mesh = Poly3DCollection(verts[faces], linewidths=0.2, alpha=1)
                    mesh.set_edgecolor('k')
                    if contact == 0:
                         mesh.set_facecolor('w')
                    if contact != 0:
                         mesh.set_facecolor(cellColorDict[contact])
                    ax.add_collection3d(mesh)
            plt.show()
    
        
        def MakeContactMovie(tracked_cell_dict, contactDict, meshDict, cell):
            
            """
            Returns a movie of a single cell over all timepoints, with the surface color coded (randomly, with each initialization) according to neighbor contact, with accompanying graph of contact area
            takes 4 parameters:
            Main tracked cell dictionary (for centroid calculation) (dict)
            Dictionary of all cell contacts (dict
            Mesh dictionary of all contact surfaces (dict)
            A cell to track (int)
            
            an array of contacts is automatically loaded from the contactArray saved to disk
            
            
            """
            
            max_contact = 0
            max_range = 30
            cont_list, cont_max = DDN.Utils.GetAllCellContacts(contactDict, cell)

            contactArray = np.load('contactArray.npy')
            
            if cell in meshDict.keys():
                for timepoint in meshDict[cell].keys():
                    if timepoint < 100:
                        #fig = plt.figure(figsize=(10, 10))
                        fig = plt.figure(figsize=plt.figaspect(0.5))
                        fig.suptitle('Surface Contact Graph of cell' + str(cell))
                        
                        ax = fig.add_subplot(1, 2, 1)
                        ax.set_xlim(0, 51)
                        ax.set_ylim(0,700)
                        ax.set_ylabel('Contact Surface Area')
                        for i in cont_list:
                            temp = contactArray[cell,i,0:timepoint] 
                            ax.plot(temp, color=cellColorDict[i])
                        
                        ax = fig.add_subplot(1, 2, 2, projection='3d')
                        centroid = tracked_cell_dict[cell][timepoint]['Centroid']
                        ax.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
                        ax.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
                        ax.set_zlim(centroid[2] - max_range, centroid[2] + max_range)
                        for contact in meshDict[cell][timepoint].keys():
                            verts = meshDict[cell][timepoint][contact]['verts']
                            faces = meshDict[cell][timepoint][contact]['faces']
                            mesh = Poly3DCollection(verts[faces], linewidths=0.2, alpha=1)
                            #mesh.set_edgecolor(cellColorDict[contact])
                            mesh.set_edgecolor('k')
                            if contact == 0:
                                mesh.set_facecolor('w')
                            if contact != 0:
                                mesh.set_facecolor(cellColorDict[contact])
                            #mesh.set_facecolor('w')
                            ax.add_collection3d(mesh)    
                        plt.tight_layout()
                        savefile = 'output\\surface\\xsurface_test_cell' + str(cell) + '_tp' + str(timepoint) + ".png"
                        #plt.show()
                        plt.savefig(savefile,bbox_inches='tight')
                        plt.close(fig)
                    


    class Interpolation:

        """Functions for interpolating missing data
        """

        def InterpolateSingleFrameCellFromDict(track, missingframe, dict):

            coords_before = dict[track][missingframe-1]['Coordinates']
            coords_after = dict[track][missingframe+1]['Coordinates']


            img_before = DDN.Utils.makeImageFromCoordinates(coords_before)
            img_after = DDN.Utils.makeImageFromCoordinates(coords_after)
            interpolated = np.zeros(shape)

            for i in range (0, 154):
                interpolated[i] = interp_shape(img_before[i],img_after[i], 0.5)

            data = interpolated >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props = regionprops(out)

            #from the props we need to return Area, Centroid and Coordinates
            Area = props[0]['Area']
            Centroid = props[0]['Centroid']
            Coordinates = props[0]['Coordinates']

            dict[track][missingframe] = {}
            dict[track][missingframe]['Area'] = Area
            dict[track][missingframe]['Centroid'] = Centroid
            dict[track][missingframe]['Coordinates'] = Coordinates
            cell_track[missingframe] = 999
            dict[track]['Track'] = cell_track
            print('Fixed  track ' + str(track) + ' at timepoint ' + str(missingframe))

            return dict


        def InterpolateTwoFramesFromDict(track, missingframe1, missingframe2):

            coords_before = tracked_cell_dict[track][missingframe1-1]['Coordinates']
            coords_after = tracked_cell_dict[track][missingframe2+1]['Coordinates']


            img_before = makeImageFromCoordinates(coords_before)
            img_after = makeImageFromCoordinates(coords_after)
            interpolated_middle = np.zeros(shape)


            #saveImageFile(img_before, 'output\\t\\1.tif')
            #saveImageFile(img_after, 'output\\t\\4.tif')

            for i in range (0, 154):
                interpolated_middle[i] = interp_shape(img_before[i],img_after[i], 0.5)


            data = interpolated_middle >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props = regionprops(out)

            interpolated_middle_img = makeImageFromCoordinates(props[0]['Coordinates'])

            interpolated_1 = np.zeros(shape)
            interpolated_2 = np.zeros(shape)

            for i in range (0, 154):
                interpolated_1[i] = interp_shape(img_before[i],interpolated_middle_img[i], 0.5)

            for i in range (0, 154):
                interpolated_2[i] = interp_shape(interpolated_middle_img[i],img_after[i], 0.5)


            data = interpolated_1 >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props1 = regionprops(out)

            #saveImageFile(out, 'output\\t\\2.tif')

            data = interpolated_2 >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props2 = regionprops(out)

            #saveImageFile(out, 'output\\t\\3.tif')



            #from the props we need to return Area, Centroid and Coordinates
            Area = props1[0]['Area']
            Centroid = props1[0]['Centroid']
            Coordinates = props1[0]['Coordinates']

            tracked_cell_dict[track][missingframe1] = {}
            tracked_cell_dict[track][missingframe1]['Area'] = Area
            tracked_cell_dict[track][missingframe1]['Centroid'] = Centroid
            tracked_cell_dict[track][missingframe1]['Coordinates'] = Coordinates
            cell_track[missingframe1] = 999
            tracked_cell_dict[track]['Track'] = cell_track


            Area = props2[0]['Area']
            Centroid = props2[0]['Centroid']
            Coordinates = props2[0]['Coordinates']

            tracked_cell_dict[track][missingframe2] = {}
            tracked_cell_dict[track][missingframe2]['Area'] = Area
            tracked_cell_dict[track][missingframe2]['Centroid'] = Centroid
            tracked_cell_dict[track][missingframe2]['Coordinates'] = Coordinates
            cell_track[missingframe2] = 999
            tracked_cell_dict[track]['Track'] = cell_track

            print('Fixed  track ' + str(track) + ' at timepoint ' + str(missingframe1) + ' '+ str(missingframe2))


    class Tracking:

        def CalculateAverageDisplacementBetweenFrames(dict, frame1, frame2):
            """
            returns the average displacement of all centroids between two frames (assuming objects are tracked in a dict). Used to refine tracking procedure to get the displacement. In this function, the dict is [timepoint][cell]
            """
            displacement_array_z = []
            displacement_array_y = []
            displacement_array_x = []
            for cell1 in dict[frame1].keys():
                centroid_frame1 = dict[frame1][cell1]['Centroid']
                min_dist, label, centroid_frame2 = findClosestCentroid(cell1, frame1, frame2)
                displacement_array_z.append(centroid_frame1[0] - centroid_frame2[0])
                displacement_array_y.append(centroid_frame1[1] - centroid_frame2[1])
                displacement_array_x.append(centroid_frame1[2] - centroid_frame2[2])

            print('average z displacement = ' + str(np.average(displacement_array_z)))
            print('average y displacement = ' + str(np.average(displacement_array_y)))
            print('average x displacement = ' + str(np.average(displacement_array_x)))
            return np.average(displacement_array_z),np.average(displacement_array_y), np.average(displacement_array_x)

        def findClosestCentroid(cell, timepoint1, timepoint2):
            centroid1 = cell_dict[timepoint1][cell]['Centroid']

            min_dist = 9999
            label = -1
            centroid = -1
            for cell2 in cell_dict[timepoint2].keys():
                centroid2 = cell_dict[timepoint2][cell2]['Centroid']
                distance = calculateDistance(np.array(centroid1), np.array(centroid2))
                if distance > 0:
                    if distance < min_dist:
                        min_dist = distance
                        label = cell2
                        centroid = cell_dict[timepoint2][cell2]['Centroid']
            return min_dist, label, centroid

        def findClosestCentroidWithDisplacement(cell, timepoint1, timepoint2, disp = (0, 0, 0)):


            centroid1 = cell_dict[timepoint1][cell]['Centroid']

            centroid1 = tuple(map(operator.add, centroid1, disp))

            min_dist = 9999
            label = -1
            centroid = -1
            for cell2 in cell_dict[timepoint2].keys():
                if 'Centroid' in cell_dict[timepoint2][cell2]:
                        centroid2 = cell_dict[timepoint2][cell2]['Centroid']
                        distance = calculateDistance(np.array(centroid1), np.array(centroid2))
                        if distance > 0:
                            if distance < min_dist:
                                min_dist = distance
                                label = cell2
                                centroid = cell_dict[timepoint2][cell2]['Centroid']
            return min_dist, label, centroid

        def TrackSingleCell(cell, start_timepoint):

            cell_track = np.zeros(51)

            cell_track[start_timepoint] = cell
            cell_label = cell
            #cell_label = cell_dict[0][28]['Old_Label']

            for i in range(start_timepoint, 50):
                #print('running cell ' + str(cell) + ' timepoint ' + str(i) + ' finding ' + str(cell_track[i]))
                if cell_track[i] > 0:
                    min_dist, label, centroid = findClosestCentroidWithDisplacement(cell_track[i], i, i+1, disp = (0, -3, 0))

                    if min_dist <= 10:
                        cell_track[i+1] = label

                    elif min_dist > 10:  #if there isn't a cell within 10 units, check the next timepoint
                        cell_track[i+1] = -1
                        if i < 49:
                            min_dist, label, centroid = findClosestCentroidWithDisplacement(cell_track[i], i, i+2, disp = (0, -6, 0))
                            if min_dist < 10:
                                cell_track[i+2] = label


            return cell_track

        def MatchNuclei(nuc_prop, labs):
            """ Take a labelled nuclear prop, and match the coordinates to a labelled cell image
                nuc_prop: a properties object  generated by regionprops of a nucleus
                labs: a labelled cell image to match the nucleus to
                returns: a cell label which the nucleus is matched to, or -1 if not label is found
            """
            matched_label = -1
            if nuc_prop['Area'] > 10:
                empty_list = []
                for coordinate in nuc_prop['Coordinates']:
                    empty_list.append(labs[coordinate[0], coordinate[1], coordinate[2]])
                matched_label = mode(empty_list)
            return matched_label


    class Vector:

        def GetEllipsoid(P):
            ET = DDN.EllipsoidTool()
            (center, radii, rotation) = ET.getMinVolEllipse(P, .01)
            return center, radii, rotation

        def GetVector(center, radii, rotation):
            """Get the center, radii and rotation of an ellipsoid from center, radii and rotation, and return:
            x, y, z: a sampled set of x y z coordinates for plotting ellipsoid
            vector_z, vector_y, vector_z: XYZ positions of the end points of a vector defining the long axis\
            """

            u = np.linspace(0.0, 2.0 * np.pi, 30)
            v = np.linspace(0.0, np.pi, 30)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

            #make a smaller subset to use for calculation
            u_s = np.linspace(0.0, 2.0 * np.pi, 10)
            v_s = np.linspace(0.0, np.pi, 10)

            x_s = np.outer(np.cos(u_s), np.sin(v_s))
            y_s = np.outer(np.sin(u_s), np.sin(v_s))
            z_s = np.outer(np.ones_like(u_s), np.cos(v_s))

            x_s = radii[0] * np.outer(np.cos(u_s), np.sin(v_s))
            y_s = radii[1] * np.outer(np.sin(u_s), np.sin(v_s))
            z_s = radii[2] * np.outer(np.ones_like(u_s), np.cos(v_s))


            for i in range(len(x_s)):
                for j in range(len(x_s)):
                    [x_s[i,j],y_s[i,j],z_s[i,j]] = np.dot([x_s[i,j],y_s[i,j],z_s[i,j]], rotation) + center

            z_list = []
            x_list = []
            y_list = []
            for i in range(len(z_s)):
                for j in range(len(z_s)):
                    z_list.append(z_s[i,j])

            for i in range(len(x_s)):
                for j in range(len(x_s)):
                    x_list.append(x_s[i,j])

            for i in range(len(y_s)):
                for j in range(len(y_s)):
                    y_list.append(y_s[i,j])

            xyz = np.column_stack((x_list,y_list,z_list))
            xyz = np.array(xyz)

            max_dist = 0
            for coord1 in xyz:
                for coord2 in xyz:
                    dist = DDN.Utils.calculateDistance(coord1, coord2)
                    if dist > max_dist:
                        max_dist = dist
                        max_coord1 = coord1
                        max_coord2 = coord2

            vector_x = [max_coord1[0], max_coord2[0]]
            vector_y = [max_coord1[1], max_coord2[1]]
            vector_z = [max_coord1[2], max_coord2[2]]


            return(x, y, z, vector_x, vector_y, vector_z)


    class EllipsoidTool:
        """Some stuff for playing with ellipsoids"""
        def __init__(self): pass

        def getMinVolEllipse(self, P=None, tolerance=0.01):
            """ Find the minimum volume ellipsoid which holds all the points

            Based on work by Nima Moshtagh
            http://www.mathworks.com/matlabcentral/fileexchange/9542
            and also by looking at:
            http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
            Which is based on the first reference anyway!

            Here, P is a numpy array of N dimensional points like this:
            P = [[x,y,z,...], <-- one point per line
                 [x,y,z,...],
                 [x,y,z,...]]

            Returns:
            (center, radii, rotation)

            """
            (N, d) = np.shape(P)
            d = float(d)

            # Q will be our working array
            Q = np.vstack([np.copy(P.T), np.ones(N)])
            QT = Q.T

            # initializations
            err = 1.0 + tolerance
            u = (1.0 / N) * np.ones(N)

            # Khachiyan Algorithm
            while err > tolerance:
                V = np.dot(Q, np.dot(np.diag(u), QT))
                M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
                j = np.argmax(M)
                maximum = M[j]
                step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
                new_u = (1.0 - step_size) * u
                new_u[j] += step_size
                err = np.linalg.norm(new_u - u)
                u = new_u

            # center of the ellipse
            center = np.dot(P.T, u)

            # the A matrix for the ellipse
            A = linalg.inv(
                           np.dot(P.T, np.dot(np.diag(u), P)) -
                           np.array([[a * b for b in center] for a in center])
                           ) / d

            # Get the values we'd like to return
            U, s, rotation = linalg.svd(A)
            radii = 1.0/np.sqrt(s)

            return (center, radii, rotation)

        def getEllipsoidVolume(self, radii):
            """Calculate the volume of the blob"""
            return 4./3.*np.pi*radii[0]*radii[1]*radii[2]

        def plotEllipsoid(self, center, radii, rotation, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
            """Plot an ellipsoid"""
            make_ax = ax == None
            if make_ax:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

            u = np.linspace(0.0, 2.0 * np.pi, 100)
            v = np.linspace(0.0, np.pi, 100)

            # cartesian coordinates that correspond to the spherical angles:
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            # rotate accordingly
            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

            if plotAxes:
                # make some purdy axes
                axes = np.array([[radii[0],0.0,0.0],
                                 [0.0,radii[1],0.0],
                                 [0.0,0.0,radii[2]]])
                # rotate accordingly
                for i in range(len(axes)):
                    axes[i] = np.dot(axes[i], rotation)


                # plot axes
                for p in axes:
                    X3 = np.linspace(-p[0], p[0], 100) + center[0]
                    Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                    Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                    ax.plot(X3, Y3, Z3, color=cageColor)

            # plot ellipsoid
            ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)

            if make_ax:
                plt.show()
                plt.close(fig)
                del fig

    class OneShot:
        def RefineBoundaries(dict, i):

            labs = np.load('cells_tracked\\timepoint_' + str(timepoint) + '.npy')
    #centroid = tracked_cell_dict[90][7]['Centroid']

            for track in dict.keys():
                clear_output(wait=True)
                print('running track ' + str(track) + ' of timepoint ' + str(timepoint))
                if timepoint in dict[track]:
                    coords = dict[track][timepoint]['Coordinates']
                    indices_to_delete = []
                    for item in range(0, len(coords)):
                        coord = coords[item]
                        if isPixelEdgeWithOtherLab(labs, coord[0], coord[1], coord[2]):
                            indices_to_delete.append(item)
                            labs[coord[0], coord[1], coord[2]] = 0
                    coords_new = np.delete(coords, indices_to_delete, axis=0)
                    dict[track][timepoint]['Coordinates'] = coords_new

            return dict

             
