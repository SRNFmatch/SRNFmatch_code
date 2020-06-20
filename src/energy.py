import torch
from pykeops.torch import Kernel, kernel_product, Genred
from pykeops.torch.kernel_product.formula import *


def Comp_normal(F, V):

    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
    N =  .5 * torch.cross(V1 - V0, V2 - V0)

    return N


def enr_match_SRNF(VS, FS, FunS, BS, VT, FT, FunT, weight_coef_dist=1 ,weight_SRNF = 1, weight_MCV = 0, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss = lossVarifoldSurf(FS, FunS, VT, FT, FunT, K)
    nS = Comp_normal(FS, VS)
    if weight_MCV>0:
        #HS = sqvmcv(FS,VS)
        HS = SRCF(FS,BS,VS)

    def energy(x):
        nx= Comp_normal(FS, x)
        if weight_MCV>0:
            Hx = SRCF(FS,BS,x)
            return weight_SRNF*SRNF_cost(nS,nx) + weight_coef_dist*dataloss(x) + weight_MCV*((Hx-HS)**2).sum(axis=1).sum()
        else:
            return weight_SRNF*SRNF_cost(nS,nx) + weight_coef_dist*dataloss(x)
          #  return weight_coef_dist*dataloss(x)
    return energy


def enr_match_SRNF_sym(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_SRNF = 1, weight_MCV = 0, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol, VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)
    def energy(x,y):
        nx= Comp_normal(F_sol, x)
        ny= Comp_normal(F_sol, y)
        if weight_MCV>0:
            #Hx = sqvmcv(F_sol,x)
            Hx = SRCF(F_sol,B_sol,x)
            #mc_x = (Hx**2).sum(axis=1).sqrt()
            #Hy = sqvmcv(F_sol,y)
            Hy = SRCF(F_sol,B_sol,y)
            #mc_y = (Hy**2).sum(axis=1).sqrt()
            return weight_SRNF*SRNF_cost(nx,ny)+ weight_MCV*((Hx-Hy)**2).sum(axis=1).sum() + weight_coef_dist_S*dataloss_S(x) + weight_coef_dist_T*dataloss_T(y)
        else:
            return weight_SRNF*SRNF_cost(nx,ny) + weight_coef_dist_S*dataloss_S(x) + weight_coef_dist_T*dataloss_T(y)
    return energy




def enr_invert_SRNF(F,Q):
# Input connectivity matrix F and target SRVF Q.
    def energy(x):
        nX= Comp_normal(F, x)
        normX = torch.norm(nX,p=2,dim=1).view(nX.shape[0],1)
        SRNF=nX/normX.sqrt()
        return ((SRNF - Q)**2).sum()
    return energy




def SRNF_cost(nX,nY):

    normX = torch.norm(nX,p=2,dim=1).view(nX.shape[0],1)
    normY = torch.norm(nY,p=2,dim=1).view(nY.shape[0],1)

    return ((nX/normX.sqrt() - nY/normY.sqrt())**2).sum()


# VT: vertices coordinates of target surface,
# FS,FT : Face connectivity of source and target surfaces
# K kernel
def lossVarifoldSurf(FS,FunS, VT, FT, FunT, K):
    def CompCLNn(F, V, Fun):

        if F.shape[1] == 2:
            V0, V1 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1])
            Fun0, Fun1 = Fun.index_select(0, F[:, 0]), Fun.index_select(0, F[:, 1])
            C, N, Fun_F =  (V0 + V1)/2, V1 - V0, (Fun0 + Fun1)/2
        else:
            V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
            Fun0, Fun1, Fun2 = Fun.index_select(0, F[:, 0]), Fun.index_select(0, F[:, 1]), Fun.index_select(0, F[:, 2])
            C, N, Fun_F =  (V0 + V1 + V2)/3, .5 * torch.cross(V1 - V0, V2 - V0), (Fun0 + Fun1 + Fun2)/3

        L = (N ** 2).sum(dim=1)[:, None].sqrt()
        [n]=list(Fun_F.size())
        Fun_F=Fun_F.resize_((n,1))
        return C, L, N / L, Fun_F

    CT, LT, NTn, Fun_FT = CompCLNn(FT, VT, FunT)
    cst = (LT * K(CT, CT, NTn, NTn, Fun_FT, Fun_FT, LT)).sum()

    def loss(VS):
        CS, LS, NSn, Fun_FS = CompCLNn(FS, VS, FunS)
        return cst + (LS * K(CS, CS, NSn, NSn, Fun_FS, Fun_FS, LS)).sum() - 2 * (LS * K(CS, CT, NSn, NTn, Fun_FS, Fun_FT, LT)).sum()

    return loss

def VKerenl(kernel_geom,kernel_grass,kernel_fun,sig_geom,sig_grass,sig_fun):
    #kernel on spatial domain
    if kernel_geom.lower() == "gaussian":
        expr_geom = 'Exp(-SqDist(x,y)*a)'
    elif kernel_geom.lower() == "cauchy":
        expr_geom = 'IntCst(1)/(IntCst(1)+SqDist(x,y)*a)'

    #kernel on Grassmanian
    if kernel_grass.lower() == 'constant':
            expr_grass = 'IntCst(1)'

    elif kernel_grass.lower() == 'linear':
            expr_grass = '(u|v)'

    elif kernel_grass.lower() == 'gaussian_oriented':
            expr_grass = 'Exp(IntCst(2)*b*((u|v)-IntCst(1)))'

    elif kernel_grass.lower() == 'binet':
            expr_grass = 'Square((u|v))'

    elif kernel_grass.lower() == 'gaussian_unoriented':
            expr_grass='Exp(IntCst(2)*b*(Square((u|v))-IntCst(1)))'

    #kernel on signal
    if kernel_fun.lower() == 'constant':
        expr_fun = 'IntCst(1)'
    elif kernel_fun.lower() == "gaussian":
        expr_fun = 'Exp(-SqDist(g,h)*c)'
    elif kernel_fun.lower() == "cauchy":
        expr_fun = 'IntCst(1)/(IntCst(1)+SqDist(g,h)*c)'



    def K(x, y, u, v, f, g, weight_y):
        d = x.shape[1]
        pK = Genred(expr_geom + '*' + expr_grass + '*' + expr_fun +'*r',
            ['a=Pm(1)','b=Pm(1)','c=Pm(1)','x=Vi('+str(d)+')','y=Vj('+str(d)+')','u=Vi('+str(d)+')',
            'v=Vj('+str(d)+')','g=Vi(1)','h=Vj(1)','r=Vj(1)'],
            reduction_op='Sum',
            axis=1)
        return pK(1/sig_geom**2,1/sig_grass**2,1/sig_fun**2,x,y,u,v,f,g,weight_y)

    return K

def SRCF(F,B,V):
    Vno = V.shape[0]
    # compute the vector area of each face
    normalOfFace = Comp_normal(F, V)

    # compute the scalar area of each face by taking a norm
    AreaOfFace = torch.norm(normalOfFace,p=2,dim=1)

    # compute the unit normal vector of each face
    # the view command appends a singleton dimension
    UnitNormalVectorOfFace = normalOfFace / AreaOfFace.reshape(AreaOfFace.shape[0],1)

    # distribute the scalar area to vertices
    AreaOfVertex = torch.zeros(V.shape[0]).to(dtype=V.dtype, device=V.device)
    AreaOfVertex.index_add_(0,F[:,0],AreaOfFace)
    AreaOfVertex.index_add_(0,F[:,1],AreaOfFace)
    AreaOfVertex.index_add_(0,F[:,2],AreaOfFace)
    AreaOfVertex.mul_(1/3)

    # compute the vector-valued mean curvature at each vertex
    # in words: for each face, for each vertex of the face,
    # compute the cross product of the opposite edge and the vector area,
    # and sum up all of these terms individually for each vertex
    VectorMeanCurvatureOfVertex = torch.zeros(V.shape).to(dtype=V.dtype, device=V.device)
    VectorMeanCurvatureOfVertex.index_add_(0,F[:,0],torch.cross(V[F[:,2],:]-V[F[:,1],:], UnitNormalVectorOfFace))
    VectorMeanCurvatureOfVertex.index_add_(0,F[:,1],torch.cross(V[F[:,0],:]-V[F[:,2],:], UnitNormalVectorOfFace))
    VectorMeanCurvatureOfVertex.index_add_(0,F[:,2],torch.cross(V[F[:,1],:]-V[F[:,0],:], UnitNormalVectorOfFace))
    VectorMeanCurvatureOfVertex.mul_(1/(8*AreaOfVertex.sqrt().reshape(Vno,1)))

    #Set the curvature vectors at boundary vertices to be 0
    VectorMeanCurvatureOfVertex[B,:] = 0

    return VectorMeanCurvatureOfVertex
