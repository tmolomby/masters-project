import csv
from coopr.pyomo import *
from coopr.opt import SolverFactory
import numpy as np
from numpy import *
import time
import MySQLdb as sql
# import matplotlib.pyplot as plt

ng=203
nb=10
nn=5
nl=6
MPC=12900
CVP=1000
relaxFlag=1
lfFlag=1
nceFlag=1
csvFlag=0
startDateTime='2011-11-01 04:30:00'
endDateTime='2011-12-01 04:00:00'
path='G:\\Masters Project\\RESULTS_2012_06\\'
path_root='G:\\Masters Project\\'
cxn = sql.connect(host="127.0.0.1", port=3306, user="root", passwd="unsw", db="results")
cursor = cxn.cursor()
s_ss_duids=['AGLHAL','AGLSOM','BARCALDN','BARRON-1','BARRON-2','BASTYAN','BBTHREE1','BBTHREE2','BBTHREE3','BDL01','BDL02','BLOWERNG','BLUFF1','BRAEMAR1','BRAEMAR2','BRAEMAR3','BRAEMAR5','BRAEMAR6','BRAEMAR7','BW01','BW02','BW03','BW04','CALL_B_1','CALL_B_2','CETHANA','CG1','CG2','CG3','CG4','CLEMGPWF','COLNSV_1','COLNSV_2','COLNSV_3','COLNSV_4','COLNSV_5','CPP_3','CPP_4','CPSA','DARTM1','DDPS1','DEVILS_G','DRYCGT1','DRYCGT2','DRYCGT3','EILDON1','EILDON2','ER01','ER02','ER03','ER04','FISHER','GORDON','GSTONE1','GSTONE2','GSTONE3','GSTONE4','GSTONE5','GSTONE6','GUNNING1','GUTHEGA','HALLWF1','HALLWF2','HUMENSW','HUMEV','HVGTS','HWPS1','HWPS2','HWPS3','HWPS4','HWPS5','HWPS6','HWPS7','HWPS8','JBUTTERS','JLA01','JLA02','JLA03','JLA04','JLB01','JLB02','JLB03','KAREEYA1','KAREEYA2','KAREEYA3','KAREEYA4','KPP_1','LADBROK1','LADBROK2','LAVNORTH','LD01','LD02','LD03','LD04','LEM_WIL','LI_WY_CA','LK_ECHO','LKBONNY2','LKBONNY3','LOYYB1','LOYYB2','LYA1','LYA2','LYA3','LYA4','MACARTH1','MACKAYGT','MACKNTSH','MCKAY1','MEADOWBK','MINTARO','MM3','MM4','MOR1','MOR2','MOR3','MORTLK11','MORTLK12','MP1','MP2','MPP_1','MPP_2','MSTUART1','MSTUART2','MSTUART3','MURRAY','NBHWF1','NPS','NPS1','NPS2','OAKEY1','OAKEY2','OAKLAND1','OSB-AG','PLAYB-AG','POAT110','POAT220','POR01','POR03','PPCCGT','QPS1','QPS2','QPS3','QPS4','QPS5','REDBANK1','REECE1','REECE2','ROMA_7','ROMA_8','SHGEN','SNOWTWN1','SNUG1','STAN-1','STAN-2','STAN-3','STAN-4','SWAN_B_1','SWAN_B_3','SWAN_E','TALWA1','TARONG#1','TARONG#2','TARONG#3','TARONG#4','TARRALEA','TNPS1','TORRA1','TORRA2','TORRA3','TORRA4','TORRB1','TORRB2','TORRB3','TORRB4','TREVALLN','TRIBUTE','TUMUT3','TUNGATIN','TVCC201','TVPP104','UPPTUMUT','URANQ11','URANQ12','URANQ13','URANQ14','VP5','VP6','VPGS','W/HOE#1','W/HOE#2','WATERLWF','WKIEWA1','WKIEWA2','WOODLWN1','WW7','WW8','YABULU','YABULU2','YWPS1','YWPS2','YWPS3','YWPS4']

if csvFlag:
    totaldemand = genfromtxt(path+'totaldemand.csv', delimiter=',')
    totaldemand = np.nan_to_num(totaldemand)
else:
    cursor.execute("SELECT * FROM totaldemand WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    totaldemand = cursor.fetchall()
    totaldemand = np.asarray(totaldemand)
    totaldemand = totaldemand[:,1:]

finish=len(totaldemand[:,1])

NCE_LHS = genfromtxt(path+'NCE_LHS.csv',delimiter=',')
nce=NCE_LHS.shape[0]
if relaxFlag:
    NCE_LHS_mod=zeros((nce,nn+ng*nb+nn+2*nl+nce))
else:
    NCE_LHS_mod=zeros((nce,nn+ng*nb+nn+2*nl))
NCE_LHS_mod[:,0:nn]=NCE_LHS[:,0:nn]
NCE_LHS_mod[:,nn:nn+ng*nb]=tile(NCE_LHS[:,nn:nn+ng],(1,nb))
NCE_LHS_mod[:,nn+ng*nb+nn:nn+ng*nb+nn+nl]=NCE_LHS[:,nn+ng:nn+ng+nl]
NCE_LHS_mod[:,nn+ng*nb+nn+nl:nn+ng*nb+nn+2*nl]=-NCE_LHS[:,nn+ng:nn+ng+nl]

NCE_RHS = genfromtxt(path+'NCE_RHS.csv',delimiter=',')
if relaxFlag:
    NCE_RHS_mod=zeros((nce,nn+ng*nb+nn+2*nl+nce))
else:
    NCE_RHS_mod=zeros((nce,nn+ng*nb+nn+2*nl))
NCE_RHS_mod[:,0:nn]=NCE_RHS[:,0:nn]
NCE_RHS_mod[:,nn:nn+ng*nb]=tile(NCE_RHS[:,nn:nn+ng],(1,nb))
NCE_RHS_mod[:,nn+ng*nb+nn:nn+ng*nb+nn+nl]=NCE_RHS[:,nn+ng:nn+ng+nl]
NCE_RHS_mod[:,nn+ng*nb+nn+nl:nn+ng*nb+nn+2*nl]=-NCE_RHS[:,nn+ng:nn+ng+nl]

NCE_constant = genfromtxt(path+'NCE_constant.csv',delimiter=',')

model = ConcreteModel()
if relaxFlag:
    N=linspace(1,ng*nb+nn+2*nl+nce,ng*nb+nn+2*nl+nce)
else:
    N=linspace(1,ng*nb+nn+2*nl,ng*nb+nn+2*nl)
model.x = Var(N, within=NonNegativeReals)
opt = SolverFactory("cbc")

def con_rule(model, m):
    if m<=nn:
        return con_lhs[int(m)-1] == float(totaldemand[t,m-1])
    elif m<=nn+ng:
        return con_lhs[int(m)-1] <= float(maxavail[t,m-nn-1])
    elif m<=nn+ng+nw:
        return con_lhs[int(m)-1] == float(totalcleared_wind[t,m-nn-ng-1])
    elif m<=nn+ng+nw+nh:
        return con_lhs[int(m)-1] <= float(totalcleared_hydro[t,m-nn-ng-nw-1])
    else:
        return con_lhs[int(m)-1] <= float(NCE_constant[m-nn-ng-nw-nh-1])-float(dot(NCE_LHS_mod[m-nn-ng-nw-nh-1,0:nn],totaldemand[t,:]))+float(dot(NCE_RHS_mod[m-nn-ng-nw-nh-1,0:nn],totaldemand[t,:]))

def con_rule_lf(model, m):
    if m<=nn:
        return sum(float(Aeq_sd_lf[m-1,i-1])*model.x[i] for i in N) == float(totaldemand[t,m-1])
    elif m<=nn+ng:
        return con_lhs[int(m)-1] <= float(maxavail[t,m-nn-1])
    elif m<=nn+ng+nw:
        return con_lhs[int(m)-1] == float(totalcleared_wind[t,m-nn-ng-1])
    elif m<=nn+ng+nw+nh:
        return con_lhs[int(m)-1] <= float(totalcleared_hydro[t,m-nn-ng-nw-1])
    else:
        return con_lhs[int(m)-1] <= float(NCE_constant[m-nn-ng-nw-nh-1])-float(dot(NCE_LHS_mod[m-nn-ng-nw-nh-1,0:nn],totaldemand[t,:]))+float(dot(NCE_RHS_mod[m-nn-ng-nw-nh-1,0:nn],totaldemand[t,:]))

def init_rule(model, i):
    if i<=ng*nb:
        return 0
    elif i<=ng*nb+nn:
        return float(totaldemand[t,i-ng*nb-1])
    else:
        return 0

if csvFlag:
    priceband1 = genfromtxt(path+'priceband1.csv', delimiter=',')
    priceband2 = genfromtxt(path+'priceband2.csv', delimiter=',')
    priceband3 = genfromtxt(path+'priceband3.csv', delimiter=',')
    priceband4 = genfromtxt(path+'priceband4.csv', delimiter=',')
    priceband5 = genfromtxt(path+'priceband5.csv', delimiter=',')
    priceband6 = genfromtxt(path+'priceband6.csv', delimiter=',')
    priceband7 = genfromtxt(path+'priceband7.csv', delimiter=',')
    priceband8 = genfromtxt(path+'priceband8.csv', delimiter=',')
    priceband9 = genfromtxt(path+'priceband9.csv', delimiter=',')
    priceband10 = genfromtxt(path+'priceband10.csv', delimiter=',')
    bandavail1 = genfromtxt(path+'bandavail1.csv', delimiter=',')
    bandavail2 = genfromtxt(path+'bandavail2.csv', delimiter=',')
    bandavail3 = genfromtxt(path+'bandavail3.csv', delimiter=',')
    bandavail4 = genfromtxt(path+'bandavail4.csv', delimiter=',')
    bandavail5 = genfromtxt(path+'bandavail5.csv', delimiter=',')
    bandavail6 = genfromtxt(path+'bandavail6.csv', delimiter=',')
    bandavail7 = genfromtxt(path+'bandavail7.csv', delimiter=',')
    bandavail8 = genfromtxt(path+'bandavail8.csv', delimiter=',')
    bandavail9 = genfromtxt(path+'bandavail9.csv', delimiter=',')
    bandavail10 = genfromtxt(path+'bandavail10.csv', delimiter=',')
    maxavail = genfromtxt(path+'maxavail.csv', delimiter=',')
    totalcleared = genfromtxt(path+'totalcleared.csv', delimiter=',')
    priceband1=np.nan_to_num(priceband1)
    priceband2=np.nan_to_num(priceband2)
    priceband3=np.nan_to_num(priceband3)
    priceband4=np.nan_to_num(priceband4)
    priceband5=np.nan_to_num(priceband5)
    priceband6=np.nan_to_num(priceband6)
    priceband7=np.nan_to_num(priceband7)
    priceband8=np.nan_to_num(priceband8)
    priceband9=np.nan_to_num(priceband9)
    priceband10=np.nan_to_num(priceband10)
    bandavail1=np.nan_to_num(bandavail1)
    bandavail2=np.nan_to_num(bandavail2)
    bandavail3=np.nan_to_num(bandavail3)
    bandavail4=np.nan_to_num(bandavail4)
    bandavail5=np.nan_to_num(bandavail5)
    bandavail6=np.nan_to_num(bandavail6)
    bandavail7=np.nan_to_num(bandavail7)
    bandavail8=np.nan_to_num(bandavail8)
    bandavail9=np.nan_to_num(bandavail9)
    bandavail10=np.nan_to_num(bandavail10)
    maxavail=np.nan_to_num(maxavail)
    totalcleared=np.nan_to_num(totalcleared)
else:
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband1 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband1 = cursor.fetchall()
    priceband1 = np.asarray(priceband1)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband2 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband2 = cursor.fetchall()
    priceband2 = np.asarray(priceband2)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband3 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband3 = cursor.fetchall()
    priceband3 = np.asarray(priceband3)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband4 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband4 = cursor.fetchall()
    priceband4 = np.asarray(priceband4)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband5 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband5 = cursor.fetchall()
    priceband5 = np.asarray(priceband5)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband6 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband6 = cursor.fetchall()
    priceband6 = np.asarray(priceband6)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband7 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband7 = cursor.fetchall()
    priceband7 = np.asarray(priceband7)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband8 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband8 = cursor.fetchall()
    priceband8 = np.asarray(priceband8)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband9 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband9 = cursor.fetchall()
    priceband9 = np.asarray(priceband9)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM priceband10 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    priceband10 = cursor.fetchall()
    priceband10 = np.asarray(priceband10)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail1 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail1 = cursor.fetchall()
    bandavail1 = np.asarray(bandavail1)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail2 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail2 = cursor.fetchall()
    bandavail2 = np.asarray(bandavail2)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail3 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail3 = cursor.fetchall()
    bandavail3 = np.asarray(bandavail3)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail4 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail4 = cursor.fetchall()
    bandavail4 = np.asarray(bandavail4)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail5 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail5 = cursor.fetchall()
    bandavail5 = np.asarray(bandavail5)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail6 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail6 = cursor.fetchall()
    bandavail6 = np.asarray(bandavail6)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail7 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail7 = cursor.fetchall()
    bandavail7 = np.asarray(bandavail7)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail8 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail8 = cursor.fetchall()
    bandavail8 = np.asarray(bandavail8)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail9 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail9 = cursor.fetchall()
    bandavail9 = np.asarray(bandavail9)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM bandavail10 WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    bandavail10 = cursor.fetchall()
    bandavail10 = np.asarray(bandavail10)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM maxavail WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    maxavail = cursor.fetchall()
    maxavail = np.asarray(maxavail)
    cursor.execute("SELECT `" + "`, `".join(s_ss_duids) + "` FROM totalcleared WHERE settlementdate BETWEEN '"+startDateTime+"' AND '"+endDateTime+"'")
    totalcleared = cursor.fetchall()
    totalcleared = np.asarray(totalcleared)

datafile = open(path_root+'20121224_Registration_and_Exemption_List.csv', 'r')
datareader = csv.reader(datafile)
data = []
duids = []
for row in datareader:
    if row[3]=='Generator' and row[4]=='Market' and (row[5]=='Scheduled' or row[5]=='Semi-Scheduled') and row[13] not in duids:
        data.append(row)
        duids.append(row[13])
# CREATE STATE MATRIX
if data[0][2]=='QLD1':
    stateMatrix=np.array([[1,0,0,0,0]])
elif data[0][2]=='NSW1':
    stateMatrix=np.array([[0,1,0,0,0]])
elif data[0][2]=='VIC1':
    stateMatrix=np.array([[0,0,1,0,0]])
elif data[0][2]=='SA1':
    stateMatrix=np.array([[0,0,0,1,0]])
elif data[0][2]=='TAS1':
    stateMatrix=np.array([[0,0,0,0,1]])
for i in range(1,len(data),1):
    if data[i][2]=='QLD1':
        stateMatrix=np.concatenate((stateMatrix,np.array([[1,0,0,0,0]])),axis=0)
    elif data[i][2]=='NSW1':
        stateMatrix=np.concatenate((stateMatrix,np.array([[0,1,0,0,0]])),axis=0)
    elif data[i][2]=='VIC1':
        stateMatrix=np.concatenate((stateMatrix,np.array([[0,0,1,0,0]])),axis=0)
    elif data[i][2]=='SA1':
        stateMatrix=np.concatenate((stateMatrix,np.array([[0,0,0,1,0]])),axis=0)
    elif data[i][2]=='TAS1':
        stateMatrix=np.concatenate((stateMatrix,np.array([[0,0,0,0,1]])),axis=0)
# CREATE WIND AND HYDRO FLAGS
if data[0][6]=='Hydro':
    hydroFlag=np.array([1])
else:
    hydroFlag=np.array([0])
if data[0][6]=='Wind':
    windFlag=np.array([1])
else:
    windFlag=np.array([0])
for i in range(1,len(data),1):
    if data[i][6]=='Hydro':
        hydroFlag=np.concatenate((hydroFlag,np.array([1])),axis=0)
    else:
        hydroFlag=np.concatenate((hydroFlag,np.array([0])),axis=0)
    if data[i][6]=='Wind':
        windFlag=np.concatenate((windFlag,np.array([1])),axis=0)
    else:
        windFlag=np.concatenate((windFlag,np.array([0])),axis=0)
windLogical=windFlag.astype(bool)
nw=sum(windFlag)
hydroLogical=hydroFlag.astype(bool)
nh=sum(hydroFlag)
if nceFlag:
    M=linspace(1,nn+ng+nw+nh+nce,nn+ng+nw+nh+nce)
else:
    M=linspace(1,nn+ng+nw+nh,nn+ng+nw+nh)
links = genfromtxt(path_root+'Links.csv',delimiter=',')
ln=links[1:,2:4]
UBint=links[1:,6]
lf_const=links[1:,8]
lf_dCoef=links[1:,9:14]
lf_lCoef=links[1:,14:]
ls=links[1:,3:5]
UBint=reshape(UBint,(1,2*nl))
UBint=reshape(UBint,(1,2*nl))
LBint=zeros((1,2*nl))

if relaxFlag:
    A_maxavail=np.concatenate((np.tile(np.identity(ng), (1, nb)),zeros((ng, nn+2*nl+nce))), 1)
else:
    A_maxavail=np.concatenate((np.tile(np.identity(ng), (1, nb)),zeros((ng, nn+2*nl))), 1)
Aeq_wind=A_maxavail[windLogical,:]
totalcleared_wind=totalcleared[:,windLogical]
totalcleared_hydro=totalcleared[:,hydroLogical]
A_hydro=A_maxavail[hydroLogical,:]
if relaxFlag:
    Aeq_sd=np.concatenate(((np.tile(stateMatrix.T, (1, nb))), np.identity(nn), zeros((nn,2*nl+nce))), 1)
    Aeq_sd_lf=np.concatenate(((np.tile(stateMatrix.T, (1, nb))), np.identity(nn), zeros((nn,2*nl+nce))), 1)
else:
    Aeq_sd=np.concatenate(((np.tile(stateMatrix.T, (1, nb))), np.identity(nn), zeros((nn,2*nl))), 1)
    Aeq_sd_lf=np.concatenate(((np.tile(stateMatrix.T, (1, nb))), np.identity(nn), zeros((nn,2*nl))), 1)
for k in range(2*nl):
    m=ln[k,0]
    n=ln[k,1]
    Aeq_sd[n-1,ng*nb+nn+k]=1
    Aeq_sd[m-1,ng*nb+nn+k]=-1

if relaxFlag:
    lower=zeros((1,ng*nb+nn+2*nl+nce))
else:
    lower=zeros((1,ng*nb+nn+2*nl))
for i in N:
    model.x[i].setlb(lower[0,i-1])

RRP_sim=zeros((finish,nn))
if relaxFlag:
    x_sim=zeros((finish,ng*nb+nn+2*nl+nce))
else:
    x_sim=zeros((finish,ng*nb+nn+2*nl))
con_lhs_value=zeros((finish,nce))
con_rhs_value=zeros((finish,nce))
A_relax=np.concatenate((zeros((nce,ng*nb+nn+2*nl)),-np.identity(nce)),1)
con_lhs=[0 for m in range(1,int(nn+ng+nw+nh+nce+1))]
for m in M:
    if m<=nn:
        con_lhs[int(m)-1]=sum(float(Aeq_sd[m-1,i-1])*model.x[i] for i in N)
    elif m<=nn+ng:
        con_lhs[int(m)-1]=sum(float(A_maxavail[m-nn-1,i-1])*model.x[i] for i in N)
    elif m<=nn+ng+nw:
        con_lhs[int(m)-1]=sum(float(Aeq_wind[m-nn-ng-1,i-1])*model.x[i] for i in N)
    elif m<=nn+ng+nw+nh:
        con_lhs[int(m)-1]=sum(float(A_hydro[m-nn-ng-nw-1,i-1])*model.x[i] for i in N)
    else:
        if relaxFlag:
            con_lhs[int(m)-1]=sum(float(NCE_LHS_mod[m-nn-ng-nw-nh-1,i+nn-1]-NCE_RHS_mod[m-nn-ng-nw-nh-1,i+nn-1]+A_relax[m-nn-ng-nw-nh-1,i-1])*model.x[i] for i in N)
        else:
            con_lhs[int(m)-1]=sum(float(NCE_LHS_mod[m-nn-ng-nw-nh-1,i+nn-1]-NCE_RHS_mod[m-nn-ng-nw-nh-1,i+nn-1])*model.x[i] for i in N)

start_time = time.time()
for t in range(finish):
    print(t*100/finish), "% complete"
    for l in [1,2]:
        if l==1:
            if relaxFlag:
                np.concatenate((priceband1[t,:],priceband2[t,:],priceband3[t,:],priceband4[t,:],priceband5[t,:],priceband6[t,:],priceband7[t,:],priceband8[t,:],priceband9[t,:],priceband10[t,:]),1)
                np.reshape(np.concatenate((priceband1[t,:],priceband2[t,:],priceband3[t,:],priceband4[t,:],priceband5[t,:],priceband6[t,:],priceband7[t,:],priceband8[t,:],priceband9[t,:],priceband10[t,:]),1),(1,ng*nb))
                c=np.concatenate((np.reshape(np.concatenate((priceband1[t,:],priceband2[t,:],priceband3[t,:],priceband4[t,:],priceband5[t,:],priceband6[t,:],priceband7[t,:],priceband8[t,:],priceband9[t,:],priceband10[t,:]),1),(1,ng*nb)),MPC*ones((1,nn)),zeros((1,2*nl)),CVP*ones((1,nce))),1)
            else:
                c=np.concatenate((np.reshape(np.concatenate((priceband1[t,:],priceband2[t,:],priceband3[t,:],priceband4[t,:],priceband5[t,:],priceband6[t,:],priceband7[t,:],priceband8[t,:],priceband9[t,:],priceband10[t,:]),1),(1,ng*nb)),MPC*ones((1,nn)),zeros((1,2*nl))),1)
            model.del_component('obj')
            model.obj = Objective(expr=sum(c[0,i-1]*model.x[i] for i in N))
            if relaxFlag:
                upper=np.concatenate((np.reshape(np.concatenate((bandavail1[t,:],bandavail2[t,:],bandavail3[t,:],bandavail4[t,:],bandavail5[t,:],bandavail6[t,:],bandavail7[t,:],bandavail8[t,:],bandavail9[t,:],bandavail10[t,:]),1),(1,ng*nb)),float("inf")*ones((1,nn)),UBint,float("inf")*ones((1,nce))),1)
            else:
                upper=np.concatenate((np.reshape(np.concatenate((bandavail1[t,:],bandavail2[t,:],bandavail3[t,:],bandavail4[t,:],bandavail5[t,:],bandavail6[t,:],bandavail7[t,:],bandavail8[t,:],bandavail9[t,:],bandavail10[t,:]),1),(1,ng*nb)),float("inf")*ones((1,nn)),UBint),1)
            for i in N:
                model.x[i].setub(float(upper[0,i-1]))
            model.del_component('con')
            model.con = Constraint(M, rule=con_rule)
            # model.pprint()
            instance = model.create()
            instance.dual = Suffix(direction=Suffix.IMPORT)
            # instance.pprint()
            results = opt.solve(instance)
            instance.load(results)
            # results.write()
        elif l==2 and lfFlag:
            lf=zeros((2*nl,1))
            x_link=zeros((2*nl,1))
            for k in range(2*nl):
                x_link[k,0]=instance.x[ng*nb+nn+k].value
            lf_temp=reshape(lf_const,(2*nl,1))+reshape(dot(lf_dCoef,totaldemand[t,:]),(2*nl,1))+dot(lf_lCoef,x_link)-ones((2*nl,1))
            for k in range(2*nl):
                lf[k]=max(lf_temp[k],0)
            for k in range(2*nl):
                m=ln[k,0]
                n=ln[k,1]
                Aeq_sd_lf[n-1,ng*nb+nn+k]=1-lf[k]*(1-ls[k,0])
                Aeq_sd_lf[m-1,ng*nb+nn+k]=-(1+lf[k]*ls[k,1])
            model.del_component('con')
            model.con = Constraint(M, rule=con_rule_lf)
            # model.pprint()
            instance = model.create()
            instance.dual = Suffix(direction=Suffix.IMPORT)
            # instance.pprint()
            results = opt.solve(instance)
            instance.load(results)
            # results.write()
    RRP_sim[t,0]=instance.dual.getValue(instance.con[1])
    RRP_sim[t,1]=instance.dual.getValue(instance.con[2])
    RRP_sim[t,2]=instance.dual.getValue(instance.con[3])
    RRP_sim[t,3]=instance.dual.getValue(instance.con[4])
    RRP_sim[t,4]=instance.dual.getValue(instance.con[5])

    # for iNCE in range(nce):
        # for i in N:
            # con_lhs_value[t,iNCE]=con_lhs_value[t,iNCE]+float((NCE_LHS_mod[iNCE,i-1]-NCE_RHS_mod[iNCE,i-1]+A_relax[iNCE,i-1])*instance.x[i].value)
        # con_rhs_value[t,iNCE]=float(NCE_constant[iNCE])-float(dot(NCE_LHS_mod[iNCE,0:nn],totaldemand[t,:]))+float(dot(NCE_RHS_mod[iNCE,0:nn],totaldemand[t,:]))

    # for i in N:
        # x_sim[t,i-1]=instance.x[i].value

print (time.time() - start_time)/finish, "seconds per loop"
# print(RRP_sim)
# print(x_sim)

# t = np.arange(0., finish, 1.)
# plt.plot(t, RRP_sim[:,0], 'm', t, RRP_sim[:,1], 'b', t, RRP_sim[:,2], 'c', t, RRP_sim[:,3], 'r', t, RRP_sim[:,4], 'g')
# plt.show()
# np.savetxt("con_lhs_value.csv", con_lhs_value, delimiter=",")
# np.savetxt("con_rhs_value.csv", con_rhs_value, delimiter=",")
np.savetxt("rrp_sim.csv", RRP_sim, delimiter=",")
# np.savetxt("x_sim.csv", x_sim, delimiter=",")
