# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:16:45 2020

@author: richter
"""


import itertools
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import factory as xes_importer

from pylab import *
np.random.seed(1)



#params
#################
log1 = xes_importer.import_log("BPIC15_1.xes")
minpts = 30
eps = 20
prominence = 0.3
#################




d = {}
Z = {}
remap_tids = []

all_rels = []
for trace in log1:
    tid = trace.attributes['concept:name']
    trace_act = [event['concept:name'] for event in trace]
    
        
        
        
        
    z = {}
    for (e1,e2) in list(itertools.combinations(trace, 2)):
        
        a1 = e1['concept:name']
        t1 = e1['time:timestamp']
        a2 = e2['concept:name']
        t2 = e2['time:timestamp']
        
        rel = a1+"->"+a2
        diff = (t2 - t1).total_seconds()
        if rel in d:
            d[rel].append(diff)
        else:
            d[rel] = [diff]
        z[rel] = diff
    Z[tid] = z
    
avg = {}
std = {}
ext = {}

for rel, values in d.items():
    avg[rel] = np.mean(values)
    std[rel] = np.std(values)
    ext[rel] = np.max(np.abs(values))
    

rels = list(d.keys())


test_time = []

    
# standardizing
Zstd = {}
for tid, trace in Z.items():
    vstd = {}
    for rel, value in trace.items():
        vstd[rel] = (value-avg[rel])
        if std[rel] == 0:
            vstd[rel] = 0
        else:
            vstd[rel] /= std[rel]
    Zstd[tid] = vstd
    
tids = Z.keys()

Zvectors = []

for tid in tids:
    
    temp = Zstd[tid]
    dummy = []
    for rel in rels:
        if rel in temp:
            dummy.append(temp[rel])
        else:
            dummy.append(0.0)
    Zvectors.append(dummy)
    remap_tids.append(tid)



print("Number of traces: ", len(Zvectors))
print("Number of relations: ", len(rels))


optics_instance = optics(Zvectors, eps, minpts)

optics_instance.process()

clusters = optics_instance.get_clusters()
noise = optics_instance.get_noise()

reach = pd.Series(optics_instance.get_ordering())
left = max(reach)

reach_smoothed = scipy.signal.savgol_filter(reach, 5, 3)

if(len(reach)%2 == 0):
    l = len(reach)-1
else:
    l = len(reach)-2

yhat = scipy.signal.savgol_filter(reach, l, 3)



coord1 = [np.array((x,reach_smoothed[x])) for x in range(len(reach_smoothed))]
coord2 = [np.array((x,yhat[x])) for x in range(len(yhat))]


diffy = (yhat-reach_smoothed)*yhat


yhatclip = np.clip(diffy, 0, np.max(diffy))


peaks, properties = find_peaks(yhatclip, prominence=prominence, width=minpts/2.0)


cc = {}
entropy_all = {}

outliers_list = []

clusters = [val for sublist in clusters for val in sublist]

for k in range(len(peaks)):
    left = np.ceil(properties["left_ips"][k])
    right = np.floor(properties["right_ips"][k])
    cc[k] = [clusters[l] for l in range(int(left),int(right))]
    
    outliers_list.extend(cc[k])
    cluster = [Zvectors[l] for l in cc[k]]
    
    
    
    entropies = []
    for i in range(len(rels)):
        rel = rels[i]
        e = np.std([v[i] for v in cluster])
        mean = np.mean([v[i] for v in cluster])
        
        entropies.append((e,rel, mean))
        
    entropy_all[k] = entropies
    
fig, ax1 = plt.subplots(figsize=(8,2))
#fig = plt.figure(figsize=(8, 4), dpi=dpi)

plt.plot(reach_smoothed)
plt.plot(yhat, color='red')

if len(peaks) == 1:
    xs = np.arange(int(properties["left_ips"]), int(properties["right_ips"]), 1)
    ax1.fill_between(xs, reach_smoothed[int(properties["left_ips"]):int(properties["right_ips"])], yhat[int(properties["left_ips"]):int(properties["right_ips"])], color='00', alpha=0.3)
elif len(peaks) > 1:
    for i in range(len(peaks)):
       xs = np.arange(int(properties["left_ips"][i]), int(properties["right_ips"][i]), 1)
       ax1.fill_between(xs, reach_smoothed[int(properties["left_ips"][i]):int(properties["right_ips"][i])], yhat[int(properties["left_ips"][i]):int(properties["right_ips"][i])], color='00', alpha=0.3) 
#plt.vlines(x=peaks, ymin=reach_smoothed[peaks], ymax = yhat[peaks], color = "C1")
#plt.hlines(y=reach_smoothed[peaks], xmin=properties["left_ips"], xmax=properties["right_ips"], color = "C1")
plt.title('TOAD plot')
plt.ylabel('Reachability Distance')
plt.xlabel('Traces')
plt.show()
