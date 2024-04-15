import numpy as np

def sign(value):
    if(value == 0):
        return(0)
    else:
        s = int(value/abs(value))
        return(s)

def soft_max(value):
    if(value < 0):
        return(0)
    else:
        return(value)

def soft_thresoulding(vector, thresould):
    for i in range(len(vector)):
        vector[i] = sign(vector[i]) * soft_max((abs(vector[i])-thresould))
    return(vector)

def unique(arr):
    my_list = []
    for element in arr:
        if element not in my_list:
            my_list.append(element)
    return my_list

def array_max(arr):
    index = int(0)
    max = arr[0]
    for i in range(len(arr)):
        if(arr[i] > max):
            max = arr[i]
            index = int(i)
    return(max, index)
def array_min(arr):
    index = int(0)
    min = arr[0]
    for i in range(len(arr)):
        if(arr[i] < min):
            min = arr[i]
            index = int(i)
    return(min, index)
def ConvertToLibsvmDataFormat(data,label, fname):
    print("data_shape=",data.shape,"label-len=",len(label))
    data_new = np.zeros((data.shape[0],data.shape[1]+1)).astype(str)
    print("datanew_shape=",data_new.shape)
    for i in range(data_new.shape[0]):
        data_new[i,0] = label[i]
        for j in range(data.shape[1]):
            data_new[i,j+1] = f'{j+1}' + ":" + f'{data[i,j]}'
    np.savetxt(fname, data_new, fmt="%s", delimiter=" ", newline = "\n")

def SvmClassificationByLibsvm(train_fname,test_fname):
    y_train, x_train = svm_read_problem(train_fname)
    y_test, x_test = svm_read_problem(test_fname)
    svm_model = svm_train(y_train, x_train, '-t 0')
    p_label_train, p_acc_train, p_val_train = svm_predict(y_train, x_train, svm_model)
    p_label_test, p_acc_test, p_val_test = svm_predict(y_test, x_test, svm_model)
    con_mat = confusion_matrix(y_test, p_label_test)
    return(con_mat,p_acc_train,p_acc_test)
