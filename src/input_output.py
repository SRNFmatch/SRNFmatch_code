#!/usr/bin/env python
import vtk
import numpy as np
from vtk.util import numpy_support
from scipy.io import loadmat
from scipy.io import savemat
import os
#import pymesh


def loadData(file_name):
        """Load MATLAB file or mesh file in format ply, obj, vtk...
        
        Returns numpy arrays V (vertices), F (faces) and Fun (functional values, set to 0 if inexistent)
        """
        path, extension = os.path.splitext(file_name)
        extension = extension.lower()                

        if extension==".mat":
            data = loadmat(file_name)
            V=data['V']
            F=data['F']-1
            F=F.astype('int64')
            
            if 'Fun' not in data:
                Fun=np.zeros((int(np.size(V)/3),))
                print("No functional values found: set to 0")
            else:
                Fun=data['Fun']
                Fun=np.reshape(Fun,(Fun.size,))
            
        else:        
            mesh = ReadPolyData(file_name)
            V = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()) #get vertices of the mesh as a numpy array
            F = numpy_support.vtk_to_numpy(mesh.GetPolys().GetData()) #get faces of the mesh as a numpy array
            
            if F.size != 3*mesh.GetNumberOfPolys():
                F=np.delete(F, slice(None, None, 4)) #delete number of vertices per cell values component
                
                F=F.reshape(int(len(F)/3),3)
                
                if mesh.GetPointData().GetScalars() is None:
                    Fun=np.zeros((int(np.size(V)/3),))
                    print("No functional values found: set to 0")
                else:
                    Fun=numpy_support.vtk_to_numpy(mesh.GetPointData().GetScalars())
                    Nbcomp=mesh.GetPointData().GetScalars().GetNumberOfComponents()
                    if Nbcomp>1:
                        Fun=np.mean(Fun,axis=1)
                        
        return V, F, Fun
        
        
def saveData(file_name,V,F,form,Fun=None,Col_range=None):
        """
        Save mesh given by numpy arrays of vertices (V), faces (F) and functional values (Fun) 
        either as a mat file or in format ply.

        """
#        mesh = pymesh.form_mesh(V,F)
#         if Fun is not None:
#            mesh.add_attribute("Function data");
#            mesh.set_attribute("Function data", Fun);
#            
#        pymesh.save_mesh(file_name,mesh,*mesh.get_attribute_names(),ascii=True, use_float=True)       
        
        if form=='mat':
            if Fun is None:
                savemat(file_name+".mat",{'V':V,'F':F+1}) 
            else:
                savemat(file_name+".mat",{'V':V,'F':F+1,'Fun':Fun}) 
            
        else:       
            writer = vtk.vtkPLYWriter()
            
            F2=F.flatten()
            F2=np.insert(F2,range(0, len(F2), 3),int(3),axis=0)
            polydata=createPolyData(V,F2)
            
            if Fun is not None:
                if Col_range is None:
                    Fun_sc = (255*(Fun - np.min(Fun))/np.ptp(Fun)).astype('uint8')
                    Fun_sc=np.column_stack((Fun_sc,Fun_sc,Fun_sc))
                else:
                    Fun_sc = (255*(Fun - Col_range[0])/np.ptp(Col_range)).astype('uint8')
                    Fun_sc=np.column_stack((Fun_sc,Fun_sc,Fun_sc)) 
                    
                Scalars_vtk = numpy_support.numpy_to_vtk(Fun_sc)
                Scalars_vtk.SetName("Colors") 
                Scalars_vtk.SetNumberOfComponents(3) 
                polydata.GetPointData().AddArray(Scalars_vtk) 
                polydata.GetPointData().SetActiveScalars("Colors") 
                writer.SetArrayName("Colors")        
                    
                    
            writer.SetInputData(polydata)
            writer.SetFileTypeToASCII()
            writer.SetFileName(file_name+".ply")
            writer.Write()
        
        
        
def plotMesh(shape):
        """
        Plot given shape=[vertex,faces,function].

        """
        Ftemp=shape[1].flatten()
        Ftemp=np.insert(Ftemp,range(0, len(Ftemp), 3),int(3),axis=0) 
        if len(shape)<3:
            Fun = None
        else:
            Fun = shape[2]  
            
        vtk_mesh = createPolyData(shape[0],Ftemp,Fun)
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        
    # Create a RenderWindowInteractor to permit manipulating the camera
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)

        ren.AddActor(polyDataToActor(vtk_mesh))
        ren.SetBackground(0.1, 0.1, 0.1)
    
    # enable user interface interactor
        iren.Initialize()
        renWin.Render()
        iren.Start()
        return iren, renWin 
        
        
def plotMatchingResult(source,match_target,target,matching_type,match_source=None):
        """
        Plot result of matching (middle) in comparison to the source (left) 
        and target (right) shapes.

        """
        Ftemp=source[1].flatten()
        Ftemp=np.insert(Ftemp,range(0, len(Ftemp), 3),int(3),axis=0)
        if len(source)<3:
            FunS = None
        else:
            FunS = source[2]               
        
        mesh_s = createPolyData(source[0],Ftemp,FunS) 

        Ftemp=target[1].flatten()
        Ftemp=np.insert(Ftemp,range(0, len(Ftemp), 3),int(3),axis=0)
        if len(target)<3:
            FunT = None
        else:
            FunT = target[2]               
        
        mesh_t = createPolyData(target[0],Ftemp,FunT) 

        Ftemp=match_target[1].flatten()
        Ftemp=np.insert(Ftemp,range(0, len(Ftemp), 3),int(3),axis=0)
        if len(match_target)<3:
            Funmt = None
        else:
            Funmt = match_target[2]               
        
        mesh_mt = createPolyData(match_target[0],Ftemp,Funmt)                 
        
        renWin = vtk.vtkRenderWindow()
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)   
        
               
        if matching_type == "Symmetric" and (match_source is not None):
                Ftemp=match_source[1].flatten()
                Ftemp=np.insert(Ftemp,range(0, len(Ftemp), 3),int(3),axis=0)
                if len(match_source)<3:
                    Funms = None
                else:
                    Funms = match_source[2]  
                    
                mesh_ms = createPolyData(match_source[0],Ftemp,Funms)   
                
                ren = vtk.vtkRenderer()
                renWin.AddRenderer(ren)
                ren.SetViewport(0,1/2,1/2,1)
                ren.AddActor(polyDataToActor(mesh_s)) 
                ren.ResetCamera()
                ren = vtk.vtkRenderer()
                renWin.AddRenderer(ren)
                ren.SetViewport(1/2,1/2,1,1)
                ren.AddActor(polyDataToActor(mesh_t)) 
                ren.ResetCamera()
                ren = vtk.vtkRenderer()
                renWin.AddRenderer(ren)
                ren.SetViewport(0,0,1/2,1/2)
                ren.AddActor(polyDataToActor(mesh_ms)) 
                ren.ResetCamera()
                ren = vtk.vtkRenderer()
                renWin.AddRenderer(ren)
                ren.SetViewport(1/2,0,1,1/2)
                ren.AddActor(polyDataToActor(mesh_mt)) 
                ren.ResetCamera()                
               
                style = vtk.vtkInteractorStyleTrackballCamera()
                iren.SetInteractorStyle(style)          
                renWin.Render()
                renWin.SetWindowName('First row: Source and Target. Second row: matched source and matched target')
                iren.Start()   
               
        else:       
               ren = vtk.vtkRenderer()
               renWin.AddRenderer(ren)
               ren.SetViewport(0,0,1/3,1)
               ren.AddActor(polyDataToActor(mesh_s)) 
               ren.ResetCamera()
               ren = vtk.vtkRenderer()
               renWin.AddRenderer(ren)
               ren.SetViewport(1/3,0,2/3,1)
               ren.AddActor(polyDataToActor(mesh_mt)) 
               ren.ResetCamera()
               ren = vtk.vtkRenderer()
               renWin.AddRenderer(ren)
               ren.SetViewport(2/3,0,1,1)
               ren.AddActor(polyDataToActor(mesh_t)) 
               ren.ResetCamera()
               
               style = vtk.vtkInteractorStyleTrackballCamera()
               iren.SetInteractorStyle(style)          
               renWin.Render()
               renWin.SetWindowName('Source, Match and Target')
               iren.Start()               

        return iren, renWin    
        
        
def plotGeodesic(Geod,F,Fun=None):
        """
        Plot geodesic evolution obtained as the output of matching.SRNF_geodesic.
        For visualization quality purpose, the maximum number of intermediate shapes is capped to 8.  

        """
        Nt=len(Geod)
        q,r = np.divmod(Nt,4)
        
        if Nt <= 4:
            L_ind=np.arange(Nt)
            n_hwin=Nt
            n_vwin=1
        elif Nt > 4 and Nt<=8:
            L_ind=np.arange(Nt)
            n_hwin=4
            n_vwin=2            
        else:
            L_ind=np.linspace(0,Nt-1,8)
            L_ind=np.rint(L_ind)
            L_ind=L_ind.astype(int)
            Nt=8
            n_hwin=4
            n_vwin=2             

        Ftemp=F.flatten()
        Ftemp=np.insert(Ftemp,range(0, len(Ftemp), 3),int(3),axis=0)                      
        
        renWin = vtk.vtkRenderWindow()
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)         
        
        for i in range(Nt):
            qi,ri = np.divmod(i, 4)
            mesh = createPolyData(Geod[L_ind[i]],Ftemp,Fun) 
            ren = vtk.vtkRenderer()
            renWin.AddRenderer(ren)
            ren.SetViewport(ri/n_hwin,1-(qi+1)/n_vwin,(ri+1)/n_hwin,1-qi/n_vwin)
            ren.AddActor(polyDataToActor(mesh)) 
            #ren.ResetCamera()           
       
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)          
        renWin.Render()
        renWin.SetWindowName('Geodesic trajectory')
        iren.Start()

        return iren, renWin           
        

        
def close_vtk_interactor(iren):
        iren.GetRenderWindow().Finalize()
        iren.TerminateApp()
        
      


def decimate_mesh(V,F,params,fun=None):
        """Decimate mesh by a given factor
        
        Returns deccimated arrays Vd (vertices), Fd (faces) and optionally Fund (functional values)
        """        
        factor=params['factor']
        Vol_preser=params['Vol_preser']
    
        F2=F.flatten()
        F2=np.insert(F2,range(0, len(F2), 3),int(3),axis=0)
    
        vtk_mesh = createPolyData(V,F2,fun)
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputData(vtk_mesh)
        decimate.SetTargetReduction(factor)
        decimate.SetVolumePreservation(Vol_preser)    

        if fun is not None:    
            Fun_Error_Metric=params['Fun_Error_Metric']
            Fun_weight=params['Fun_weigth']    
            decimate.SetAttributeErrorMetric(Fun_Error_Metric)
            decimate.SetScalarsWeight(Fun_weight)
    
    
        decimate.Update()

        decimated = vtk.vtkPolyData()
        decimated.ShallowCopy(decimate.GetOutput())
    
        Vd_vtk = decimated.GetPoints().GetData()
        Vd = numpy_support.vtk_to_numpy(Vd_vtk)
        Fd_vtk=decimated.GetPolys().GetData()
        Fd=numpy_support.vtk_to_numpy(Fd_vtk)
        Fd=np.delete(Fd, slice(None, None, 4))
        Fd=Fd.reshape(int(len(Fd)/3),3) 
    
        if fun is not None: 
            fund_vtk=decimated.GetPointData().GetScalars()
            fund=numpy_support.vtk_to_numpy(fund_vtk)
            return Vd, Fd, fund
        
        else:
            return Vd, Fd


def subdivide_mesh(V,F,order):
    F2=F.flatten()
    F2=np.insert(F2,range(0, len(F2), 3),int(3),axis=0)  
    vtk_mesh = createPolyData(V,F2)
    linear = vtk.vtkLinearSubdivisionFilter()
    linear.SetInputData(vtk_mesh)
    linear.SetNumberOfSubdivisions(order)
    linear.Update()
    subdivided_mesh = vtk.vtkPolyData()
    subdivided_mesh.ShallowCopy(linear.GetOutput())
    VS = numpy_support.vtk_to_numpy(subdivided_mesh.GetPoints().GetData()) #get vertices of the mesh as a numpy array
    FS = numpy_support.vtk_to_numpy(subdivided_mesh.GetPolys().GetData()) #get faces of the mesh as a numpy array
    FS=np.delete(FS, slice(None, None, 4))
    FS=FS.reshape(int(len(FS)/3),3)     
    return VS, FS    
     


def ReadPolyData(file_name):
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".g":
        reader = vtk.vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        poly_data = None
    return poly_data


def createPolyData(verts,tris,funs=None):
        """Create and return a vtkPolyData.
        
        verts is a (N, 3) numpy array of float vertices

        tris is a (N, 1) numpy array of int64 representing the triangles
        (cells) we create from the verts above.  The array contains 
        groups of 4 integers of the form: 3 A B C
        Where 3 is the number of points in the cell and A B C are indexes
        into the verts array.
        """
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(verts))        
        poly.SetPoints(points)
        cells = vtk.vtkCellArray()
        cells.SetCells(int(len(tris) / 4), numpy_support.numpy_to_vtkIdTypeArray(tris))        
        poly.SetPolys(cells)         
        if funs is not None:
            poly.GetPointData().SetScalars(numpy_support.numpy_to_vtk(funs))
        
        return poly
        
        

def polyDataToActor(polydata):
        """Wrap the provided vtkPolyData object in a mapper and an actor, returning
            the actor.
        """
        mapper = vtk.vtkPolyDataMapper()
        #    if vtk.VTK_MAJOR_VERSION <= 5:
#           #mapper.SetInput(reader.GetOutput())
#           mapper.SetInput(polydata)
#           else:
#           mapper.SetInputConnection(polydata.GetProducerPort())
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()     
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        
        return actor       