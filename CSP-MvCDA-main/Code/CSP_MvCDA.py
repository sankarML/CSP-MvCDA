import numpy as np
from sklearn import preprocessing
import scipy
from scipy.sparse.csgraph import laplacian
import pandas as pd
import math
import os
from numpy import linalg as LA
from timeit import default_timer as timer
import sys
import my_library
from libsvm.svmutil import *
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from gsindex import geometrical_separability_index



n = len(sys.argv)
if(n<2):
    print("enter the value of the parameter beta and alpha")
    print("Total arguments passed:", n)
    os.system("exit")

beta = float(sys.argv[1])
alpha = float(sys.argv[2])
#data_folder=sys.argv[3]

data_directory = "../Data/"
result_directory = "../Result/"

data_folder = "ALOI"
Modality =['VIEW_1','VIEW_2','VIEW_3','VIEW_4']

n = len(Modality)   #no of view
component = 25      #The required number of latent vectors
feature = np.zeros(n).astype(int)



def within_scatter(data_matrix,class_label):
    print(data_matrix.shape)
    sample = data_matrix.shape[0]
    feature = data_matrix.shape[1]
    ScatterW = np.zeros((feature, feature))
    class_index = my_library.unique(class_label)
    for c in class_index:
        index = []
        for i in range(sample):
            if(class_label[i] == c):
                index.append(i)
        #print("class, index =",c,index )
        class_data = data_matrix[index,:]
        class_data_cen = class_data-class_data.mean(axis = 0)
        class_variance = np.matmul(np.transpose(class_data_cen),class_data_cen)
        ScatterW = ScatterW + class_variance
    return(ScatterW)

def matrix_Inv(mat):
    print("Inverse Start")
    d,U = scipy.linalg.eigh(mat)
    d1 = d[::-1]
    total_var = np.sum(d1)
    U1 = np.zeros(U.shape)
    for i in range(mat.shape[0]):
        U1[i] = U[i][::-1]
    index = len(d1)
    for j in range(len(d1)):
        sum1 = np.sum(d1[:j])
        s = sum1/total_var
        print("variance =", s*100.0)
        r = abs(d1[j]/d1[j-1])
        if(s>0.99):
            index = j
            print("dj",d1[j],"dj-1",d1[j-1])
            break
    print("Considered_Rank=", index)
    D_Inv = np.diag((1/d1[0:index]))
    U_t = np.transpose(U1[:,0:index])
    print("u shape=",U1[:,0:index].shape)
    print("u_t shape=",U_t.shape)
    mat_Inv = LA.multi_dot([U1[:,0:index],D_Inv,U_t])
    print("Inverse End")
    return(mat_Inv)


def model_score(X,x_latent):
    x_score = np.matmul(X,x_latent)
    return(x_score)

def feature_to_array(feature):
    sum = 0
    l = len(feature)+1
    arr=np.zeros(l).astype(int)
    for i in range(l-1):
        sum = sum+feature[i]
        arr[i+1]=sum
    return sum,arr

def ClassWiseGraphLaplacian(class_label,sample):
    Graph_W = np.zeros((sample,sample)).astype(int)
    Graph_B = np.zeros((sample,sample)).astype(int)
    Class_ID = np.unique(class_label)
    for k in Class_ID:
        index = np.where(class_label==k)[0]
        for i in index:
            for j in index:
                if(j != i):
                    Graph_W[i,j]=1
    for i in range(sample):
        for j in range(sample):
            if(j != i):
                if(Graph_W[i,j] == 0):
                    Graph_B[i,j]=1
                else:
                    Graph_B[i,j]=0
    Lap_graphW = laplacian(Graph_W)
    Lap_graphB = laplacian(Graph_B)
    return Lap_graphW,Lap_graphB

def Discriminate_matrix(Lap_graphW,Lap_graphB,X,alpha):
    temp1 = LA.multi_dot([np.transpose(X),Lap_graphW,X])
    temp2 = LA.multi_dot([np.transpose(X),Lap_graphB,X])
    result = alpha*temp1-(1-alpha)*temp2
    return result

def Make_path_print(data_folder, result_directory, Algo, para_folder=""):
    new_path = os.path.join(result_directory,data_folder)
    if(os.path.exists(new_path) == False):
        os.mkdir(new_path)
    new_path = os.path.join(new_path,Algo)
    if(os.path.exists(new_path) == False):
        os.mkdir(new_path)
    if(len(para_folder)!=0):
        new_path = os.path.join(new_path,para_folder)
        if(os.path.exists(new_path) == False):
            os.mkdir(new_path)

def SvmClassificationByLibsvm(train_fname,test_fname):
    y_train, x_train = svm_read_problem(train_fname)
    y_test, x_test = svm_read_problem(test_fname)
    svm_model = svm_train(y_train, x_train, '-t 0')
    p_label_train, p_acc_train, p_val_train = svm_predict(y_train, x_train, svm_model)
    p_label_test, p_acc_test, p_val_test = svm_predict(y_test, x_test, svm_model)
    con_mat = confusion_matrix(y_test, p_label_test)
    return(con_mat,p_acc_train,p_acc_test)


def proposed_method(Data,feature, class_label, n, component):
    Sxw={}
    Dx = {}
    Sxw_Inv={}
    cross_cov={}
    Latent_vec={}
    a={}
    a_new={}
    T,arr = feature_to_array(feature)

    print("Laplacian matrix calculation: start")
    Lap_GraphW,Lap_GraphB = ClassWiseGraphLaplacian(class_label,len(class_label))
    print("Laplacian matrix calculation: end")

    print("within scatter and discrimation matrix calculation: start")
    for i in range(n):
        Sxw[i] = within_scatter(Data[i], class_label)
        Dx[i]= Discriminate_matrix(Lap_GraphW,Lap_GraphB,Data[i],alpha)
    print("within scatter nd discrimation matrix calculation is done")

    print("Inverse calculation: start")
    for i in range(n):
        Sxw_Inv[i] = matrix_Inv(Sxw[i])
    print("Inverse calculation is done")

    print("Cross covariance calculation for all pair: start")
    for i in range(n):
        cross_cov[i]={}
        for j in range(n):
            if(i==j):
                cross_cov[i][j] = (-1)*beta*Dx[i]
            else:
                cross_cov[i][j] = np.matmul(np.transpose(Data[i]), Data[j])
    temp_mat={}
    for i in range(n):
        temp_mat[i]={}
        for j in range(n):
                temp_mat[i][j]=np.matmul(Sxw_Inv[i],cross_cov[i][j])

    print("Cross covariance calculation is done")
    for i in range(n):
        Latent_vec[i]=np.zeros((feature[i],component))
        #print("l",i,Latent_vec[i].shape)
    DinvR = np.zeros((T,T))
    for i in range(n):
        for j in range(n):
            DinvR[arr[i]:arr[i+1], arr[j]:arr[j+1]] = temp_mat[i][j]

    #np.savetxt("DINVR",DinvR,fmt="%s", delimiter=' ', newline='\n')
    eval, evec = scipy.sparse.linalg.eigsh(DinvR, k=component)
    eval1 = eval[::-1]
    print("eigen_value",eval1)
    evec1 = np.zeros(evec.shape)
    for i in range(T):
        evec1[i] = evec[i][::-1]
    for l in range(n):
        print(Latent_vec[l].shape)
        Latent_vec[l]=evec1[arr[l]:arr[l+1],:]
    return(Latent_vec)


Algo = "CSP-MvCDA"
para_folder = "beta_"+str(beta) +"alpha_"+str(alpha)
print('Dataset:',data_folder)
print("----------------------------------------------------------------------------")
print("Patameters: beta--",beta,"alpha--",alpha)
print("Training Testing dataset is used")
print("----------------------------------------------------------------------------")
accuracy = np.zeros(2)
x_train_cen={}
x_test_cen={}
x_TrainScore={}
x_TestScore={}
gsi = np.zeros(2)


# Data Read
print("Training info")
print("----------------------------------------------------------------------")
Data_train={}
for d in range(len(Modality)):
    Data_train[d]= np.array(pd.read_csv(data_directory+data_folder+"/"+Modality[d]+"/training.txt", sep = '\t',header = None,skiprows=1))[:,:-1]
    feature[d] = Data_train[d].shape[1]
    print("Modality:", Modality[d],"and shape:", Data_train[d].shape)
class_label_train=np.array(pd.read_csv(data_directory+data_folder+"/"+Modality[d]+"/training.txt", sep = '\t',header = None,skiprows=1))[:,-1].astype(int)
print("Class label:",class_label_train)

print("Test info")
print("----------------------------------------------------------------------")
Data_test={}
for d in range(len(Modality)):
    Data_test[d]=np.array(pd.read_csv(data_directory+data_folder+"/"+Modality[d]+"/test.txt", sep = '\t',header = None,skiprows=1))[:,:-1]
    feature[d] = Data_test[d].shape[1]
    print("modality:", Modality[d],"and shape:", Data_test[d].shape)
class_label_test=np.array(pd.read_csv(data_directory+data_folder+"/"+Modality[d]+"/test.txt", sep = '\t',header = None,skiprows=1))[:,-1].astype(int)
print("Class label:",class_label_test)


# For Centered Data (zero mean)
# for i in range(n):
#     x_train_cen[i] = Data_train[i] - Data_train[i].mean(axis = 0)
#     #x_test_cen[i] = Data_test[i] - Data_test[i].mean(axis = 0)
#     x_test_cen[i] = Data_test[i] - Data_train[i].mean(axis = 0)


# For Standardised Data (zero mean one variance)
for i in range(n):
    scaler = StandardScaler().fit(Data_train[i])
    x_train_cen[i] = scaler.transform(Data_train[i])
    x_test_cen[i] = scaler.transform(Data_test[i])


start_time = timer()
Latent_vector = proposed_method(x_train_cen,feature,class_label_train,n,component)
end_time = timer()
print("Latent Vectors are calculated")

#print(Latent_vector)
sum1=np.zeros((Data_train[0].shape[0],component))
sum2=np.zeros((Data_test[0].shape[0],component))
for j in range(n):
    x_TrainScore[j]=model_score(x_train_cen[j],Latent_vector[j])
    sum1=sum1+ x_TrainScore[j]
    x_TestScore[j]=model_score(x_test_cen[j],Latent_vector[j])
    sum2=sum2+ x_TestScore[j]


print("train_feature_shape:",sum1.shape)
print("test_feature_shape:",sum2.shape)

# Geometrical Separablity Index
gsi[0] = geometrical_separability_index(sum1, class_label_train)
gsi[1] = geometrical_separability_index(sum2, class_label_test)
print("GSI Train: ", gsi[0], "GSI Test: ", gsi[1])


Make_path_print(data_folder, result_directory, Algo, para_folder)
for i in range(n):
    file1 = "Latent_view"+f'{i}'+'.txt'
    file2 = "Train_score_view"+f'{i}'+'.txt'
    file3 = "Test_score_view"+f'{i}'+'.txt'
    np.savetxt(os.path.join(result_directory,data_folder,Algo,para_folder,file1), Latent_vector[i], fmt='%s', delimiter=' ', newline='\n')
    np.savetxt(os.path.join(result_directory,data_folder,Algo,para_folder,file2), x_TrainScore[i], fmt='%s', delimiter=' ', newline='\n')
    np.savetxt(os.path.join(result_directory,data_folder,Algo,para_folder,file3), x_TestScore[i], fmt='%s', delimiter=' ', newline='\n')

train_fname = os.path.join(result_directory,data_folder,Algo,para_folder,"data_train.txt")
test_fname = os.path.join(result_directory,data_folder,Algo,para_folder,"data_test.txt")
my_library.ConvertToLibsvmDataFormat(sum1,class_label_train,train_fname)
my_library.ConvertToLibsvmDataFormat(sum2,class_label_test,test_fname)
conf_mat,train_accuracy,test_accuracy = SvmClassificationByLibsvm(train_fname,test_fname)
accuracy[0] = train_accuracy[0]
accuracy[1] = test_accuracy[0]
f = os.path.join(result_directory,data_folder,Algo,para_folder)
np.savetxt(f+"/accuracy.txt",accuracy, fmt = "%s", delimiter="\t")
np.savetxt(f+"/confusion_matirx.txt",conf_mat, fmt = "%s", delimiter="\t",newline="\n")
np.savetxt(f+"/exe_time.txt",np.array([end_time-start_time]), fmt = "%s", delimiter="\t",newline="\n")
np.savetxt(os.path.join(result_directory,data_folder,Algo,para_folder,"gs_index.txt"),gsi, fmt = "%s", delimiter="\t")
