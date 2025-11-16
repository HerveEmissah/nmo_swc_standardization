# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 23:24:37 2020

@author: NANDA
"""
import os
#import sys
#import glob
#import cPickle as pickle
import numpy as np
#import matplotlib.pyplot as plt
import re
        
def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                #list_of_results.append((line_number, line.rstrip()))
                list_of_results.append(line_number)
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results


def makedescending(uM):
    newPid = uM[:,6].astype(int)
    index = uM[:,0].astype(int)
    k=1
    for npid in uM[1:,6]:
        x = np.where(index==npid)
        newPid[k] = int(x[0])+1        
        if k<len(newPid):
            k = k+1
    uuM = uM[:]
    uuM[:,0] = range(1, len(uM[:,1])+1) 
    uuM[:,6] = newPid
    return uuM

#find parent branch
def find_parent_side_branch(lineNumb1, fln):
    indx = fln[:,0].astype(int)
    parid = fln[:,6].astype(int)
    Br = [];
    BrDaut = [];
    for ind in indx:
        B = np.where(parid==ind)
        if len (B[0])>1:
            BrDaut.append(B[0])
            Br.append(ind)
    lineNumb2 = []
    for li in lineNumb1:
        if not parid[li] in Br:
            intmedp = parid[li]-1
            lineNumb2.append(intmedp)
            while parid[intmedp] not in Br:
                intmed = parid[intmedp]-1
                lineNumb2.append(intmed)
                intmedp = intmed
    return lineNumb2

#function to remove non tip lines falsely identified as side branches. 
def remove_non_tips(lineN1,fln):
    parid = fln[:,6].astype(int)
    d = 0
    de = []
    for ln in lineN1:
        if (ln+1) in parid:
            de.append(d)
        d = d+1
    lineN2 = np.delete(lineN1,de , axis=0)
    return lineN2         
    

#beginnning of delete side branch
def delete_side_branch(file_name, lineNumbers):
    os.chdir('../CNG_Version')
    file_name_swc = file_name[:len(file_name)-4]
    with open(file_name_swc) as flr:
            fls = flr.read().splitlines()
    #ignore the lines that start with "#" and " "
    prefixes = ('#');
    for word in fls[:]:
        if word.startswith(prefixes):
            fls.remove(word)
    #remove blank lines        
    while("" in fls) : 
        fls.remove("")
#    fls_1 = [string for string in fls if string != ""]
#    fls = fls_1
#    del fls_1
    i=0
    #convert the string list to an np array
    fl=np.zeros((len(fls),7),float)

#    for fll in fls:
#        X = fll.split()
#        X = np.array([X])
#        #X = X.astype(np.float)
#        X = X.astype(float)
#        if len(X[0,:])==6:
#            X=np.append([X], [-1])
#        fl[i,:]= X 
#        i=i+1

    for i, fll in enumerate(fls):
        # Split the line into individual values
        X = np.array(fll.split(), dtype=float)
    
        # If the length is 6, append -1 to make it 7
        if len(X) == 6:
            X = np.append(X, -1)
    
        # Assign the values to the array
        fl[i, :] = X
   
    #delete variables
    del X, fls, i, word
     #now remove the side branch single points one by one 
    #work of the lines                
    lineNum1 = np.array(lineNumbers)
    for l in range(0,len(lineNum1)):
        lineNum1[l]  = int(lineNum1[l]) -1  
    lineNum1 = lineNum1.astype(int)
    lineNum1 = remove_non_tips(lineNum1, fl)
    lineNum2 = find_parent_side_branch(lineNum1, fl)
    if len(lineNum2)>0:
        lineNum2 = lineNum2.astype(int) 
        lineNum1.append(lineNum2)
    fldel = np.delete(fl,lineNum1 , axis=0) 
    fldelCon = makedescending(fldel)
    os.chdir("..")           
    if not os.path.exists("out"):
        os.mkdir("out")
    os.chdir("out")      
    np.savetxt(file_name_swc,fldelCon,fmt="%u %u %f %f %f %f %d")
#    file_name_swc_swc = (file_name_swc + ".swc")
#    np.savetxt(file_name_swc_swc,fldel,fmt="%u %u %f %f %f %f %d")
    os.chdir("../Remaining_issues")
#end of delete side branch
            
  
#main script
# Get the default list of variables
#save_list = dir()
# get the file names
Erlist = search_string_in_file('Log.txt', '2.7=>')
ELi = np.array(Erlist)
ELi = np.insert(ELi, 0, 0., axis=0)

with open('Log.txt') as f:
    ids = f.read().splitlines()
substr = 'swc.std'
fnam = []
for i in range(1,len(ELi)) :
    for k in range(ELi[i]-1,ELi[i-1],-1):
        if substr in ids[k]:
            fnam.append(ids[k])
            break
        
del ELi, Erlist, i, ids, k, substr
## add fnam to list of variables not to be deleted
#save_list.append([fnam])
## delete the rest of the variables
#for name in dir():
#    if name not in save_list:
#        del globals()[name]
#
#for name in dir():
#    if name not in save_list:
#        del locals()[name]
         
os.chdir('Remaining_issues')
pattern = "2.7  Line (.*?) of"
for fn in fnam:
    with open(fn) as f:
        Lines = f.read().splitlines()
    fids = search_string_in_file(fn, '2.7  Line')  
    lnum = []
    for fi in fids:
        #fid = int(np.array(search_string_in_file(fn, '2.7 ')))-1
        str1 = Lines[fi-1]
        lineNum = re.search(pattern, str1).group(1)
        #lineNum = map(int, re.findall(r'\d+', str1))
        #lineNum = lineNum[2]
        lnum.append(lineNum)
        #print(f"fi in {fn}: {fi}")  # Print the value of fi in each iteration
    try:    
        delete_side_branch(fn, lnum)
    except:
        os.chdir("..\Remaining_issues")
        pass
    del Lines, fi, fids, lineNum, lnum, str1
    #clear variables 
    
            
            
        
        