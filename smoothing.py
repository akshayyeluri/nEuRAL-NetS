import numpy as np
import pandas as pd
from time import time
from scipy.stats import zscore
from scipy.stats import itemfreq
np.set_printoptions(threshold=np.nan, suppress=True)
pd.set_option('display.width', 1000)

def label_freq(y):
    # func returns frequcy of labels as list (#0's, #1's, #-1's)
    item_freq = itemfreq(y)    
    label_freq = [item_freq[1,1], item_freq[2,1], item_freq[0,1]]
    return label_freq

def quote_data_labels(data):
    
    #check how many time multiple transactions occur at same time
    #i.e. multiple data rows with same t
    t_vals, t_counts = np.unique(data[:,0], return_counts=True)
    t_dup_trans = np.sum(t_counts[t_counts>1])
    print ("{} instances with multiple transactions at same time:".format(t_dup_trans))
    
    #get price i.e. mid-quotes
    quote = (data[:,6] + data[:,8])*0.5
    
    #get price diff i.e. p(t+1) - p(t) 
    labels = np.diff(quote)
    labels[labels>0] = 1
    labels[labels<0] = -1
    
    # np.diff returns array with one less element, so add 0 for last row / latest time
    labels = np.append(labels, 0)
    #data = data[:-1]
    
    ones = (labels[labels==1].size)/labels.size
    negs = (labels[labels==-1].size)/labels.size
    zeros = 1 - negs - ones
    print ("Labels Generated 0:1:-1 = {}:{}:{}".format(round(zeros*100,2), round(ones*100,2), round(negs*100,2)))   
  
    return data, labels



    
def Order_Data(X):
    
    rows = X[:,0].size
    X_ord = np.array(X[:,:6])  # all order columns, remove LOB columns
    X_ord_time = np.array(X_ord[:,0]).reshape((rows,1)) # only time column
    X_ord_type = np.array(X_ord[:,1]).reshape((rows,1))
    X_ord_dir = np.array(X_ord[:,5]).reshape((rows,1))
    X_ord_id = np.array(X_ord[:,2]).reshape((rows,1))
    X_ord_det = np.array(X_ord[:,3:6])
    
    return_list = [('Orders-All', X_ord), ('Order-Time', X_ord_time), 
                   ('Order-Type', X_ord_type), ('Order-Dir', X_ord_dir), 
                   ('Order-ID', X_ord_id), ('Order-Details', X_ord_det)]

    return return_list

def LOB_Depths(X):
   
    XT = np.delete(X,[1,2,3,4,5],axis=1)  # remove messages columns
    XT10 = XT[:,-40:]
    XT5 = XT[:,-20:]
    XT4 = XT[:,-16:]
    XT3 = XT[:,-12:]
    XT2 = XT[:,-8:]
    XT1 = XT[:,-4:]
    #print (XT.shape, XT10.shape, XT5.shape, XT1.shape)
    
    return_list = [('LOB-10', XT10), ('LOB-5', XT5), 
                   ('LOB-4', XT4), ('LOB-3', XT3), 
                   ('LOB-2', XT2), ('LOB-1', XT1)]
                  
    return return_list

def LOB_Imbalance(X):
    XT = np.delete(X,[0,1,2,3,4,5],axis=1)  # remove messages columns
    
    X_time = X[:,0].reshape((X[:,0].size,1))
    
    # bid - ask [Gould, Bonart 2015]
    XT10_diff = XT[:,[3,7,11,15,19,23,27,31,35,39]] - XT[:,[1,5,9,13,17,21,25,29,33,37]] 
    XT10_sum = XT[:,[3,7,11,15,19,23,27,31,35,39]] + XT[:,[1,5,9,13,17,21,25,29,33,37]]
    XT10 = XT10_diff / XT10_sum
    
    #print (XT10_diff[10], XT10_sum[10], XT10[10])
    
    XT10 = np.concatenate((X_time, XT10), axis=1)
    XT5 = np.concatenate((X_time, XT10[:,:5]), axis=1)
    XT4 = np.concatenate((X_time, XT10[:,:4]), axis=1)
    XT3 = np.concatenate((X_time, XT10[:,:3]), axis=1)
    XT2 = np.concatenate((X_time, XT10[:,:2]), axis=1)
    XT1 = np.concatenate((X_time, XT10[:,:1]), axis=1)
    
    #print (XT10.shape, XT5.shape, XT4.shape, XT3.shape, XT2.shape, XT1.shape)

    return_list = [('IMB-10', XT10), ('IMB-5', XT5), 
                   ('IMB-4', XT4), ('IMB-3', XT3), 
                   ('IMB-2', XT2), ('IMB-1', XT1)]
                  
    return return_list
 
def Normalize_Data (X, X_imb):
    # calculate zscores for X, except for columns 0,1,2,5
    XT = np.array(X)
    XT = zscore(XT)
    XT[:,0] = X[:,0]
    XT[:,1] = X[:,1]
    XT[:,2] = X[:,2]
    XT[:,5] = X[:,5]
    
    #claculate zscores for all cols of imbalance dataset
    XT_imb = np.array(X_imb)
    XT_imb = zscore(XT_imb) 

    return XT, XT_imb
     
def Order_Arrival_Rates(X, y, t_thresh=5):
    # This function sums the order creations, cancels and executions 
    # for a given historic window (t_thresh)
    # XT cols = Create Vol Buy, Create Vol Sell, Canc Vol Buy, Canc Vol Sell, Exec Vol
    
    fn_start = X[0,0]
    cutoff_t = fn_start + t_thresh 
        
    XT = np.array(X[(X[:,0]>cutoff_t)][:,:14])
    yT = np.array(y[(X[:,0]>cutoff_t)])
    #print (X.shape, XT.shape, y.shape, yT.shape)
    XT[:,1:14] = 0
    
    t1 = time()

    for i, row in enumerate (XT):
        t_end = XT[i,0]
        t_start = t_end - t_thresh
        
        win = np.array(X[(X[:,0]>t_start) & (X[:,0]<t_end)][:,:])
        
        if not win[:,0].size == 0: 
        
            create_vol_buy = np.sum(win[(win[:,1] == 1) & (win[:,5] == 1)][:,3])
            create_vol_sell = np.sum(win[(win[:,1] == 1) & (win[:,5] == -1)][:,3])
            create_vol_diff = create_vol_buy - create_vol_sell
            
            create_vol_buy_lob = np.sum(win[(win[:,1] == 1) & (win[:,5] == 1) & (win[:,4] >= win[:,8])][:,3])
            create_vol_sell_lob = np.sum(win[(win[:,1] == 1) & (win[:,5] == -1) & (win[:,4] <= win[:,6])][:,3])
            create_vol_diff_lob = create_vol_buy_lob - create_vol_sell_lob      
            
            canc_vol_buy = np.sum(win[((win[:,1] == 2) | (win[:,1] == 3)) & (win[:,5] == 1)][:,3])
            canc_vol_sell = np.sum(win[((win[:,1] == 2) | (win[:,1] == 3)) & (win[:,5] == -1)][:,3])
            canc_vol_diff = canc_vol_buy - canc_vol_sell
            
            canc_vol_buy_lob = np.sum(win[((win[:,1] == 2) | (win[:,1] == 3)) & (win[:,5] == 1) & (win[:,4] >= win[:,8])][:,3])
            canc_vol_sell_lob = np.sum(win[((win[:,1] == 2) | (win[:,1] == 3)) & (win[:,5] == -1) & (win[:,4] <= win[:,6])][:,3])
            canc_vol_diff_lob = canc_vol_buy_lob - canc_vol_sell_lob
             
            exec_vol = np.sum(win[(win[:,1] == 4) | (win[:,1] == 5)][:,3])    
            
            #print (create_vol_buy, create_vol_sell)
            #print (canc_vol_buy, canc_vol_sell)
            #print (exec_vol)
                
            XT[i,1:] = [create_vol_buy, create_vol_sell, create_vol_diff, 
                        canc_vol_buy, canc_vol_sell, canc_vol_diff, exec_vol,
                        create_vol_buy_lob, create_vol_sell_lob, create_vol_diff_lob, 
                        canc_vol_buy_lob, canc_vol_sell_lob, canc_vol_diff_lob]   
    
    XT_rate_ords = np.array(XT[:,0:8])
    XT_rate_lobords = np.array(XT[:,(1,8,9,10,11,12,13)])
    
    print ("Vrate Data Setup Time", time()-t1)
    
    return XT, XT_rate_ords, XT_rate_lobords, yT
        
def Combine_Tranformed_Data (X, X_imb, X_arrt_1, y):
    
    fn_start = X[0,0]
    fn_end = X[-1,0]
    
    cutoff_start = fn_start + 30
    cutoff_end = fn_end -30
    
    X_ind_st = X[X[:,0]<cutoff_start][:,0].size
    X_ind_end = X[X[:,0]>cutoff_end][:,0].size
    
    Xarrt_ind_st = X_arrt_1[X_arrt_1[:,0]<cutoff_start][:,0].size
    Xarrt_ind_end = X_arrt_1[X_arrt_1[:,0]>cutoff_end][:,0].size
    
    print (X_ind_st,X_ind_end,Xarrt_ind_st,Xarrt_ind_end)
    
    X1 = np.array(X[X_ind_st:-X_ind_end,:])
    X2 = np.array(X_imb[X_ind_st:-X_ind_end,:])
    X3 = np.array(X_arrt_1[Xarrt_ind_st:-Xarrt_ind_end,:])
    
    print (X1[:,0].size,X2[:,0].size,X3[:,0].size)
    yT =  np.array(y[X_ind_st:-X_ind_end])
    
    XT = np.concatenate ((X1, X2, X3), axis=1)

    return XT, yT
    
    
def LOB_Change_Rates(X, y, t_thresh=1):
    
    # XT cols = Create Vol Buy, Create Vol Sell, Canc Vol Buy, Canc Vol Sell, Exec Vol
    '''
    fn_start = X[0,0]
    cutoff_t = fn_start + t_thresh 
        
    XT = np.concatenate((X[:,0],np.array(X[(X[:,0]>cutoff_t)][:,6:])), axis=1)
    yT = np.array(y[(X[:,0]>cutoff_t)])
    print (X.shape, XT.shape, y.shape, yT.shape)
    #XT[:,:] = 0
    
    t1 = time()

    for i, row in enumerate (XT):
        t_end = XT[i,0]
        t_start = t_end - t_thresh
        
        win = np.array(X[(X[:,0]>t_start) & (X[:,0]<t_end)][:,6:])
        
        if not win[:,0].size == 0: 
    
            diff = win[-1] - win[0] 
            XT[i,1:] = diff
            #print (create_vol_buy, create_vol_sell)
            #print (canc_vol_buy, canc_vol_sell)
            #print (exec_vol)
                
    print ("LOB Change Rates Data Setup Time", time()-t1)
    
    print (cutoff_t, XT.shape, yT.shape, XT[:2,:])
    '''
    return ('Lob-Rate', [], [])
