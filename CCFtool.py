#!/usr/bin/env python
"""
# Author: Juanru Guo
# Created Time : 
# File Name: 
# Description:
"""
from xmlrpc.server import list_public_methods
import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import lil_matrix
import anndata as ad
from scipy.sparse import csr_matrix
from numba import jit
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Mapping, Literal
from matplotlib.axes import Axes
from matplotlib import pyplot as pl
from matplotlib import rcParams, cm
from matplotlib.axes import Axes

@jit(nopython=True)
def _findinsertionslen2(Chrom, start, end, length = 3, startpoint = 0,totallength = 10000000):
    # function to calculate the number of hops in the spcific area of chromosomes
    
    count = 0
    initial = startpoint
    flag = 0
    
    for i in range(startpoint,totallength):
        if Chrom[i] >= start-length and Chrom[i] <= end :
            if flag == 0:
                initial = i
                flag = 1
            count += 1
        elif Chrom[i] > end and count!= 0:
            return count,initial
        
    return count,initial


def _findinsertionslen(Chrom, start, end, length = 3):
    
    # function to calculate the number of hops in the spcific area of chromosomes
    return len(Chrom[(Chrom >= max(start-length,0)) & (Chrom <= end )])

def _findinsertions(Chrom, start, end, length = 3):

    # function returns of hops in the spcific area of chromosomes
    return Chrom[(Chrom >= max(start-length,0)) & (Chrom <= end)]

def _compute_cumulative_poisson(exp_hops_region,bg_hops_region,total_exp_hops,total_bg_hops,pseudocounts):
    
    from scipy.stats import poisson
    
    # Calculating the probability under the hypothesis of possion distribution
    if total_bg_hops >= total_exp_hops:
        return(1-poisson.cdf((exp_hops_region+pseudocounts),bg_hops_region * (float(total_exp_hops)/float(total_bg_hops)) + pseudocounts))
    else:
        return(1-poisson.cdf(((exp_hops_region *(float(total_bg_hops)/float(total_exp_hops)) )+pseudocounts),bg_hops_region + pseudocounts))


_PeaktestMethod = Optional[Literal["poisson","binomial"]]

def testCompare_bf(
    bound: list, 
    curChromnp: np.ndarray, 
    curframe: np.ndarray, 
    length: int, 
    lam_win_size: Optional[int] ,  
    boundnew: list, 
    pseudocounts: float = 0.2, 
    pvalue_cutoff: float  = 0.00001, 
    chrom: str = None, 
    test_method: _PeaktestMethod = "poisson", 
    record: bool = True) -> list:
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
        
    # test whether the potiential peaks are true peaks by comparing to other data
    
    for i in range(len(bound)):
        
        # calculate the total number of hops in tatal
        TTAAnum = _findinsertionslen(curframe, bound[i][0], bound[i][1], length) 
        boundnum = bound[i][2]
        
        if lam_win_size == None:
            
            scaleFactor = float(len(curChromnp)/len(curframe))
            lam = TTAAnum * scaleFactor +pseudocounts
            
            if test_method == "poisson":
                pvalue = 1-poisson.cdf(boundnum , lam)
            elif test_method == "binomial":
                pvalue = binomtest(int(boundnum+pseudocounts), n=len(curChromnp), 
                                   p=((TTAAnum+pseudocounts)/len(curframe)) , alternative='greater').pvalue
                
        else:
            
            TTAAnumlam = _findinsertionslen(curframe, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, length) 
            boundnumlam = _findinsertionslen(curChromnp, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, length) 
            
  
            scaleFactor = float(boundnumlam/TTAAnumlam)
            lam = TTAAnum * scaleFactor +pseudocounts
            
            if test_method == "poisson":  
                pvalue = 1-poisson.cdf(boundnum , lam)
            elif test_method == "binomial":
                pvalue = binomtest(int(boundnum+pseudocounts), n=boundnumlam, 
                                   p=((TTAAnum+pseudocounts)/TTAAnumlam) , alternative='greater').pvalue
        
        if pvalue < pvalue_cutoff:
            if record:
                boundnew.append([chrom, bound[i][0], bound[i][1], boundnum, TTAAnum, lam, pvalue])
            else:
                boundnew.append([chrom, bound[i][0], bound[i][1]])
                
    return boundnew

def testCompare_bf2(
    bound: list, 
    curChromnp: np.ndarray, 
    curframe: np.ndarray, 
    length: int, 
    lam_win_size: Optional[int],  
    boundnew: list, 
    pseudocounts: float = 0.2, 
    pvalue_cutoff: float  = 0.00001, 
    chrom: str = None, 
    test_method: _PeaktestMethod = "poisson", 
    record: bool = True) -> list:
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
        
    # test whether the potiential peaks are true peaks by comparing to other data
    
    startpointTTAA = 0
    
    if lam_win_size != None:
        startpointTTAAlam = 0
        startpointboundlam = 0
        
        
    totallengthcurframe = len(curframe)
    totallengthcurChromnp = len(curChromnp)
    
    for i in range(len(bound)):
        
        # calculate the total number of hops in total
        TTAAnum, startpointTTAA = _findinsertionslen2(curframe, bound[i][0], bound[i][1], length, startpointTTAA ,totallengthcurframe) 
        boundnum = bound[i][2]
        
        if lam_win_size == None:
            
            scaleFactor = float(totallengthcurChromnp/totallengthcurframe)
            lam = TTAAnum * scaleFactor +pseudocounts
            
            if test_method == "poisson":
                pvalue = 1-poisson.cdf(boundnum , lam)
            elif test_method == "binomial":
                pvalue = binomtest(int(boundnum+pseudocounts), n=totallengthcurChromnp, 
                                   p=((TTAAnum+pseudocounts)/totallengthcurframe) , alternative='greater').pvalue
                
        else:
            
            TTAAnumlam,startpointTTAAlam = _findinsertionslen2(curframe, bound[i][0] - lam_win_size/2 + 1, 
                                                             bound[i][1] + lam_win_size/2, length, 
                                                             startpointTTAAlam, totallengthcurframe) 
            boundnumlam,startpointboundlam = _findinsertionslen2(curChromnp, bound[i][0] - lam_win_size/2 + 1, 
                                                               bound[i][1] + lam_win_size/2, length,
                                                              startpointboundlam, totallengthcurChromnp) 
            
         
            scaleFactor = float(boundnumlam/TTAAnumlam)
            lam = TTAAnum * scaleFactor +pseudocounts
            
            if test_method == "poisson":  
                pvalue = 1-poisson.cdf(boundnum , lam)
            elif test_method == "binomial":
                pvalue = binomtest(int(boundnum+pseudocounts), n=boundnumlam, 
                                   p=((TTAAnum+pseudocounts)/TTAAnumlam) , alternative='greater').pvalue
        
        if pvalue < pvalue_cutoff:
            if record:
                boundnew.append([chrom, bound[i][0], bound[i][1], boundnum, TTAAnum, lam, pvalue])
            else:
                boundnew.append([chrom, bound[i][0], bound[i][1]])
                
    return boundnew

def testCompare(
    bound: list, 
    curChromnp : np.ndarray,
    curbgframe: np.ndarray,
    curTTAAframenp: np.ndarray,
    length: int,
    lam_win_size: Optional[int] , 
    boundnew: list,
    pseudocounts: float, 
    pvalue_cutoffbg: float, 
    pvalue_cutoffTTAA: float, 
    chrom: str, 
    test_method: _PeaktestMethod, 
    record: bool) -> list:

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
        
    totalcurChrom = len(curChromnp)
    totalcurbackground = len(curbgframe)
    totalcurTTAA = len(curTTAAframenp)
    
    # test whether the potiential peaks are true peaks by comparing to other data
    for i in range(len(bound)):
        
        bgnum = _findinsertionslen(curbgframe, bound[i][0], bound[i][1], length) 
        TTAAnum = _findinsertionslen(curTTAAframenp,bound[i][0], bound[i][1], length) 
        boundnum = bound[i][2]
        
        
        if lam_win_size == None:
            
            totalcurChrom = len(curChromnp)
            totalcurbackground = len(curbgframe)
            totalcurTTAA = len(curTTAAframenp)
            
            scaleFactorTTAA = totalcurChrom/totalcurTTAA
            lamTTAA = TTAAnum * scaleFactorTTAA +pseudocounts
            
            scaleFactorbg = totalcurChrom/totalcurbackground
            lambg = TTAAnum * scaleFactorbg +pseudocounts
            
            if test_method == "poisson":
                
                pvalueTTAA = 1-poisson.cdf(boundnum , lamTTAA)
                pvaluebg = _compute_cumulative_poisson(boundnum,bgnum,totalcurChrom,totalcurbackground,pseudocounts)
                
            elif test_method == "binomial":
                
                pvalueTTAA = binomtest(int(boundnum+pseudocounts), n=totalcurChrom, 
                                   p=((TTAAnum+pseudocounts)/totalcurTTAA ) , alternative='greater').pvalue
                pvaluebg = binomtest(int(boundnum+pseudocounts), n=totalcurChrom, 
                                   p=((bgnum+pseudocounts)/totalcurbackground) , alternative='greater').pvalue
            
        else:
            
            bgnumlam = _findinsertionslen(curbgframe, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, length)
            TTAAnumlam = _findinsertionslen(curTTAAframenp, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, length) 
            boundnumlam = _findinsertionslen(curChromnp, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, length) 
            
            scaleFactorTTAA = bgnumlam/TTAAnumlam
            lamTTAA = TTAAnum * scaleFactorTTAA +pseudocounts
            
            scaleFactorbg = bgnumlam/boundnumlam
            lambg = TTAAnum * scaleFactorbg +pseudocounts
            
            if test_method == "poisson":
                
                pvalueTTAA = 1-poisson.cdf(boundnum , lamTTAA)
                pvaluebg = _compute_cumulative_poisson(boundnum,bgnum,boundnumlam,bgnumlam,pseudocounts)
                
            elif test_method == "binomial":
                
                pvalueTTAA = binomtest(int(boundnum+pseudocounts), n=bgnumlam, 
                                   p=((TTAAnum+pseudocounts)/TTAAnumlam ) , alternative='greater').pvalue
                pvaluebg = binomtest(int(boundnum+pseudocounts), n=bgnumlam, 
                                   p=((bgnum+pseudocounts)/boundnumlam) , alternative='greater').pvalue
   
        
        if pvaluebg < pvalue_cutoffbg and pvalueTTAA < pvalue_cutoffTTAA :
            
            if record:
                boundnew.append([chrom, bound[i][0], bound[i][1], boundnum, bgnum, TTAAnum, lambg, lamTTAA, pvaluebg, pvalueTTAA])
            else:
                boundnew.append([chrom, bound[i][0], bound[i][1]])
                
    return boundnew


def testCompare2(
    bound: list, 
    curChromnp : np.ndarray,
    curbgframe: np.ndarray,
    curTTAAframenp: np.ndarray,
    length: int,
    lam_win_size: Optional[int] ,  
    boundnew: list,
    pseudocounts: float, 
    pvalue_cutoffbg: float, 
    pvalue_cutoffTTAA: float, 
    chrom: str, 
    test_method: _PeaktestMethod, 
    record: bool) -> list:
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
    # test whether the potiential peaks are true peaks by comparing to other data
    
    startbg = 0
    startTTAA = 0
    
    totalcurChrom = len(curChromnp)
    totalcurbackground = len(curbgframe)
    totalcurTTAA = len(curTTAAframenp)
    
    if lam_win_size != None:
        
        startbglam = 0
        startTTAAlam = 0
        startboundlam = 0
        
    for i in range(len(bound)):
        
        
        bgnum, startbg = _findinsertionslen2(curbgframe, bound[i][0], bound[i][1], length, startbg, totalcurbackground ) 
        TTAAnum, startTTAA = _findinsertionslen2(curTTAAframenp,bound[i][0], bound[i][1], length, startTTAA, totalcurTTAA) 
        boundnum = bound[i][2]
        
        
        if lam_win_size == None:

            
            scaleFactorTTAA = totalcurChrom/totalcurTTAA
            lamTTAA = TTAAnum * scaleFactorTTAA +pseudocounts
            
            scaleFactorbg = totalcurChrom/totalcurbackground
            lambg = TTAAnum * scaleFactorbg +pseudocounts
            
            if test_method == "poisson":
                
                pvalueTTAA = 1-poisson.cdf(boundnum , lamTTAA)
                pvaluebg = _compute_cumulative_poisson(boundnum,bgnum,totalcurChrom,totalcurbackground,pseudocounts)
                
            elif test_method == "binomial":
                
                pvalueTTAA = binomtest(int(boundnum+pseudocounts), n=totalcurChrom, 
                                   p=((TTAAnum+pseudocounts)/totalcurTTAA ) , alternative='greater').pvalue
                pvaluebg = binomtest(int(boundnum+pseudocounts), n=totalcurChrom, 
                                   p=((bgnum+pseudocounts)/totalcurbackground) , alternative='greater').pvalue
            
        else:


            bgnumlam, startbglam = _findinsertionslen2(curbgframe, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, 
                                                      length, startbglam, totalcurbackground)
            TTAAnumlam, startTTAAlam = _findinsertionslen2(curTTAAframenp, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2,
                                                       length, startTTAAlam, totalcurTTAA) 
            boundnumlam, startboundlam = _findinsertionslen2(curChromnp, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, 
                                             length, startboundlam, totalcurChrom) 
            
            scaleFactorTTAA = bgnumlam/TTAAnumlam
            lamTTAA = TTAAnum * scaleFactorTTAA +pseudocounts
            
            scaleFactorbg = bgnumlam/boundnumlam
            lambg = TTAAnum * scaleFactorbg +pseudocounts
            
            if test_method == "poisson":
                
                pvalueTTAA = 1-poisson.cdf(boundnum , lamTTAA)
                pvaluebg = _compute_cumulative_poisson(boundnum,bgnum,boundnumlam,bgnumlam,pseudocounts)
                
            elif test_method == "binomial":
                
                pvalueTTAA = binomtest(int(boundnum+pseudocounts), n=bgnumlam, 
                                   p=((TTAAnum+pseudocounts)/TTAAnumlam ) , alternative='greater').pvalue
                pvaluebg = binomtest(int(boundnum+pseudocounts), n=bgnumlam, 
                                   p=((bgnum+pseudocounts)/boundnumlam) , alternative='greater').pvalue
   
        
        if pvaluebg < pvalue_cutoffbg and pvalueTTAA < pvalue_cutoffTTAA :
            
            if record:
                boundnew.append([chrom, bound[i][0], bound[i][1], boundnum, bgnum, TTAAnum, lambg, lamTTAA, pvaluebg, pvalueTTAA])
            else:
                boundnew.append([chrom, bound[i][0], bound[i][1]])
                
    return boundnew



def test_bf(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    pvalue_cutoff: float = 0.01,  
    mininser: int = 5, 
    minlen: int =  0,
    extend: int = 150, 
    maxbetween: int = 2800,  
    lam_win_size: Optional[int] =None,  
    pseudocounts: float = 0.2, 
    test_method: _PeaktestMethod = "poisson", 
    record: bool = False) -> pd.DataFrame:
    

    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):

        if len(curTTAAframe) == 0:
            continue
            
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))


        # make a summary of our current insertion start points
        unique, counts = np.unique(curChromnp, return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []

        # initial the start point, end point and the totol number of insertions
        startbound = 0
        endbound = 0
        insertionbound = 0

        # calculate the distance between each points
        dif1 = np.diff(unique, axis=0)
        # add a zero to help end the following loop at the end
        dif1 = np.concatenate((dif1,np.array([maxbetween+1])))


        # look for the uique insertion points
        for i in range(len(unique)):
            if startbound == 0:
                startbound = unique[i]
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i] 
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0 
            else:
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i]
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0

        
        boundnew = testCompare_bf(bound, curChromnp, curTTAAframe, length, lam_win_size,  boundnew,  pseudocounts, 
                                  pvalue_cutoff, chrom,  test_method = test_method,record = True)
        
    if record:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End", "Experiment Hops", "Background Hops", "Expected Hops", "pvalue"])

    else:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End"])

def test_bf2(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    pvalue_cutoff: float = 0.01,  
    mininser: int = 5, 
    minlen: int =  0,
    extend: int = 150, 
    maxbetween: int = 2800,  
    lam_win_size: Optional[int] =None,  
    pseudocounts: float = 0.2, 
    test_method: _PeaktestMethod = "poisson", 
    record: bool = False) -> pd.DataFrame:
    

    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):

        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        if len(curTTAAframe) == 0:
            continue
            
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        
        

        # make a summary of our current insertion start points
        unique, counts = np.unique(curChromnp, return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []

        # initial the start point, end point and the totol number of insertions
        startbound = 0
        endbound = 0
        insertionbound = 0

        # calculate the distance between each points
        dif1 = np.diff(unique, axis=0)
        # add a zero to help end the following loop at the end
        dif1 = np.concatenate((dif1,np.array([maxbetween+1])))


        # look for the uique insertion points
        for i in range(len(unique)):
            if startbound == 0:
                startbound = unique[i]
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i] 
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0 
            else:
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i]
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0

        
        boundnew = testCompare_bf2(bound, curChromnp, curTTAAframe, length, lam_win_size,  boundnew,  pseudocounts, 
                                  pvalue_cutoff, chrom,  test_method = test_method,record = True)

        
    if record:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End", "Experiment Hops", "Background Hops", "Expected Hops", "pvalue"])

    else:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End"])


    

def test(
    expdata: pd.DataFrame, 
    backgroundframe: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    pvalue_cutoffbg: float = 0.00001, 
    pvalue_cutoffTTAA: float = 0.000001,  
    mininser: int = 5, 
    minlen: int = 0,
    extend: int = 150, 
    maxbetween: int = 2800,  
    lam_win_size: Optional[int] =None,  
    pseudocounts: float = 0.2, 
    test_method:  _PeaktestMethod = "poisson", 
    record: bool = False) -> pd.DataFrame:

  
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):
        

        curbackgroundframe = np.array(list(backgroundframe[backgroundframe[0]==chrom][1]))
        if len(curbackgroundframe) == 0:
            continue
        curbackgroundframe.sort()
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        if len(curTTAAframe) == 0:
            continue
            
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChrom.sort()
        


        # make a summary of our current insertion start points
        unique, counts = np.unique(np.array(curChrom), return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []

        # initial the start point, end point and the totol number of insertions
        startbound = 0
        endbound = 0
        insertionbound = 0

        # calculate the distance between each points
        dif1 = np.diff(unique, axis=0)
        # add a zero to help end the following loop at the end
        dif1 = np.concatenate((dif1,np.array([maxbetween+1])))


        # look for the uique insertion points
        for i in range(len(unique)):
            if startbound == 0:
                startbound = unique[i]
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i] 
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0 
            else:
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i]
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0

                               
        boundnew = testCompare(bound, curChromnp, curbackgroundframe, curTTAAframe, length, lam_win_size, boundnew, pseudocounts, pvalue_cutoffbg, pvalue_cutoffTTAA, chrom, test_method , record)
 
        
    if record:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End", "Experiment Hops", 
                                                   "Background Hops", "TTAA Hops", "Expected Hops background", "Expected Hops TTAA", 
                                                       "pvalue background", "pvalue TTAA"])

    else:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End"])

    
def test2(
    expdata: pd.DataFrame, 
    backgroundframe: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    pvalue_cutoffbg: float = 0.00001, 
    pvalue_cutoffTTAA: float= 0.000001,  
    mininser: int = 5, 
    minlen: int = 0,
    extend: int = 150, 
    maxbetween: int = 2800,  
    lam_win_size: Optional[int] =None,  
    pseudocounts: float = 0.2, 
    test_method:  _PeaktestMethod = "poisson", 
    record: bool = False)-> pd.DataFrame:

    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):
        
        curbackgroundframe = np.array(list(backgroundframe[backgroundframe[0]==chrom][1]))
        if len(curbackgroundframe) == 0:
            continue
        curbackgroundframe.sort()
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        if len(curTTAAframe) == 0:
            continue

        
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        

            

        # make a summary of our current insertion start points
        unique, counts = np.unique(np.array(curChrom), return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []

        # initial the start point, end point and the totol number of insertions
        startbound = 0
        endbound = 0
        insertionbound = 0

        # calculate the distance between each points
        dif1 = np.diff(unique, axis=0)
        # add a zero to help end the following loop at the end
        dif1 = np.concatenate((dif1,np.array([maxbetween+1])))


        # look for the uique insertion points
        for i in range(len(unique)):
            if startbound == 0:
                startbound = unique[i]
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i] 
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0 
            else:
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i]
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0

                               
        boundnew = testCompare2(bound, curChromnp, curbackgroundframe, curTTAAframe, length, lam_win_size, boundnew, pseudocounts, pvalue_cutoffbg, pvalue_cutoffTTAA, chrom, test_method , record)
 
        
    if record:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End", "Experiment Hops", 
                                                   "Background Hops", "TTAA Hops", "Expected Hops background", "Expected Hops TTAA", 
                                                       "pvalue background", "pvalue TTAA"])

    else:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End"])

def blockifyCompare(
    bound: list, 
    curChrom: np.ndarray, 
    curframe: np.ndarray, 
    length: int, 
    boundnew: list, 
    scaleFactor: float, 
    pseudocounts: float, 
    pvalue_cutoff: float, 
    chrom: str, 
    test_method:  _PeaktestMethod = "poisson", 
    record: bool = True) -> list:
# test whether the potiential peaks are true peaks by comparing to TTAAs

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
    last = -1
    Chrnumtotal = 0
    TTAAnumtotal = 0

    for i in range(len(bound)):
        TTAAnum = _findinsertionslen(curframe, bound[i][0], bound[i][1], length)
        boundnum = bound[i][2]
        
        if test_method == "poisson":
            pValue = 1-poisson.cdf(boundnum - 1, TTAAnum * scaleFactor+pseudocounts)
            
        elif test_method == "binomial":
            pValue = binomtest(int(boundnum+pseudocounts), n=len(curChrom), 
                               p=((TTAAnum+pseudocounts)/len(curframe) ) , alternative='greater').pvalue
        

        if pValue <= pvalue_cutoff and last == -1 :
            
            last = i
            Chrnumtotal += boundnum
            TTAAnumtotal += TTAAnum

        elif pValue > pvalue_cutoff and last != -1 :
            
            if record:
                
                if test_method == "poisson":
                    pvalue = 1-poisson.cdf(Chrnumtotal - 1, TTAAnumtotal * scaleFactor+pseudocounts)

                elif test_method == "binomial":
                    pvalue = binomtest(int(Chrnumtotal+pseudocounts), n=len(curChrom), 
                                       p=((TTAAnumtotal+pseudocounts)/len(curframe) ) , alternative='greater').pvalue
                    
                boundnew.append([chrom, bound[last][0], bound[i-1][1], Chrnumtotal, TTAAnumtotal, TTAAnumtotal*scaleFactor+pseudocounts, pvalue])
            else:
                boundnew.append([chrom, bound[last][0], bound[i-1][1]])
                
            last = -1
            Chrnumtotal = 0
            TTAAnumtotal = 0


    if last != -1:
        
        if record:
            
            if test_method == "poisson":
                pvalue = 1-poisson.cdf(Chrnumtotal - 1, TTAAnumtotal * scaleFactor+pseudocounts)

            elif test_method == "binomial":
                pvalue = binomtest(int(Chrnumtotal+pseudocounts), n=len(curChrom), 
                                   p=((TTAAnumtotal+pseudocounts)/len(curframe) ) , alternative='greater').pvalue

            boundnew.append([chrom, bound[last][0], bound[i-1][1], Chrnumtotal, TTAAnumtotal, TTAAnumtotal*scaleFactor+pseudocounts, pvalue])
        else:
            boundnew.append([chrom, bound[last][0], bound[i-1][1]])

    return boundnew


def Blockify(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    pvalue_cutoff: float = 0.0001, 
    pseudocounts: float = 0.2, 
    test_method:  _PeaktestMethod = "poisson", 
    record: bool = True) -> pd.DataFrame:
    
    

    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):
        

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))

        if len(curTTAAframe) == 0:
            continue
            
        # make a summary of our current insertion start points
        unique, counts = np.unique(curChromnp, return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []


        import astropy.stats as astrostats
        hist, bin_edges = astrostats.histogram(expdata[expdata[0] == chrom][1], bins="blocks")

        hist = list(hist)
        bin_edges = list(bin_edges.astype(int))
        for bins in range(len(bin_edges)-1):
            bound.append([bin_edges[bins],bin_edges[bins+1], hist[bins], bin_edges[bins+1]-bin_edges[bins]])


        
        boundnew = blockifyCompare(bound, curChromnp, curTTAAframe, length, 
                                   boundnew, scaleFactor = len(curChromnp)/len(curTTAAframe), pseudocounts = pseudocounts, 
                                   pvalue_cutoff = pvalue_cutoff, chrom = chrom, test_method = test_method, record = record)

        
    if record:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End", "Experiment Hops", "Background Hops", "Expected Hops", "pvalue"])

    else:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End"])


def callpeaksMACS2_bf(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    window_size: int = 1000,
    lam_win_size: Optional[int] = 100000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    pvalue_cutoff: float = 0.01,
    record: bool = False) -> pd.DataFrame:
    
    # function for MACS2 background free
    from scipy.stats import poisson
    
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    chr_list = []
    start_list = []
    end_list = []
    
    if record:
        center_list = []
        num_exp_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        lambda_type_list =[]
        lambda_list = []
        pvalue_list = []
        l = []
        background_hops = []
        expect_hops = []
        
    total_experiment_hops = len(expdata)
    list_of_l_names = lam_win_size
    
    
    for chrom in tqdm.tqdm(chrm):
    #for chrom in chrm:

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChrom.sort()
        
        max_pos = curChrom[-1]
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        
        for window_start in range(0,int(max_pos+window_size+(lam_win_size/2+1)),step_size):
            
            num_exp_hops = _findinsertionslen(curChromnp, window_start, window_start+window_size - 1, length)
            if num_exp_hops>1:
                num_lam_win_size_hops = _findinsertionslen(curChromnp, window_start-(lam_win_size/2-1), 
                                                       window_start+window_size +(lam_win_size/2) - 1, length)
                num_TTAAs_window = _findinsertionslen(curTTAAframe, window_start, window_start+window_size - 1, length)
                num_TTAAs_lam_win_size = _findinsertionslen(curTTAAframe, window_start-(lam_win_size/2-1), 
                                                       window_start+window_size +(lam_win_size/2) - 1, length)
                lambda_lam_win_size = (num_lam_win_size_hops/(max(num_TTAAs_lam_win_size,1))) #expected number of hops per TTAA

                #is this window significant?
                pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),lambda_lam_win_size*max(num_TTAAs_window,1)+pseudocounts)
            else:
                pvalue = 1
                
            if pvalue < pvalue_cutoff :
                #was last window significant?
                if sig_flag:
                    #if so, extend end of windows
                    sig_end = window_start+window_size-1
                else:
                    #otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start+window_size-1
                    sig_flag = 1

            else:
                #current window not significant.  Was last window significant?
                if sig_flag:
                    

                    #compute peak center and add to frame
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)
            
                    #add number of experiment hops in peak to frame
                    num_exp_hops = len(overlap)
                    
                    num_TTAAs_peak = _findinsertionslen(curTTAAframe, sig_start, sig_end, length)

                    #compute lambda in lam_win_size
                    num_exp_hops_lam_win_size = _findinsertionslen(curChromnp, peak_center-(lam_win_size/2-1), peak_center+(lam_win_size/2), length)
                    num_TTAAs_lam_win_size = _findinsertionslen(curTTAAframe, peak_center-(lam_win_size/2-1), peak_center+(lam_win_size/2), length)
                    lambda_lam_win_size = float(num_exp_hops_lam_win_size)/(max(num_TTAAs_lam_win_size,1), length)


                    lambda_f = lambda_lam_win_size

                    #compute pvalue and record it

                    pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),lambda_f*max(num_TTAAs_peak,1)+pseudocounts)
                    pvalue_list.append(pvalue)              
                    #number of hops that are a user-defined distance from peak center
                    
                    if record:
                        
                        #add chr, peak start, peak end
                        chr_list.append(chrom) #add chr to frame
                        start_list.append(sig_start) #add peak start to frame
                        end_list.append(sig_end) #add peak end to frame
                        center_list.append(peak_center) #add peak center to frame
                        num_exp_hops_list.append(num_exp_hops)

                        #add fraction of experiment hops in peak to frame
                        frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                        tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)
                        #record type of lambda used
                        lambda_type_list.append(list_of_l_names)
                        #record lambda
                        lambda_list.append(lambda_f)
                        background_hops.append(num_TTAAs_peak)
                        expect_hops.append(lambda_f*max(num_TTAAs_peak,1)+pseudocounts)
                        
                    sig_flag = 0


                    
    if record:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center","Experiment Hops",
                "Fraction Experiment","TPH Experiment","Lambda Type",
                "Lambda","Poisson pvalue"])

        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Hops"] = num_exp_hops_list 
        peaks_frame["Fraction Experiment"] = frac_exp_list 
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Lambda Type"] = lambda_type_list
        peaks_frame["Lambda"] = lambda_list
        peaks_frame["Background Hops"] = background_hops
        peaks_frame["Expect Hops"] = expect_hops
        
    else:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Poisson pvalue"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["Poisson pvalue"] = pvalue_list
                    
    peaks_frame = peaks_frame[peaks_frame["Poisson pvalue"] <= pvalue_cutoff]
    
    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]
    
    
def callpeaksMACS2(
    expdata: pd.DataFrame, 
    background: pd.DataFrame,  
    TTAAframe: pd.DataFrame, 
    length: int,
    window_size: int = 1000,
    lam_win_size: Optional[int] =100000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    pvalue_cutoff: float = 0.01,
    record: bool = False) -> pd.DataFrame:
    
    # function for MACS2 with background 
    from scipy.stats import poisson
    
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    chr_list = []
    start_list = []
    end_list = []
    list_of_l_names = ["bg","1k","5k","10k"]
    pvalue_list = []
    
    if record:
        chr_list = []
        start_list = []
        end_list = []
        center_list = []
        num_exp_hops_list = []
        num_bg_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        frac_bg_list = []
        tph_bg_list = []
        tph_bgs_list = []
        lambda_type_list =[]
        lambda_list = []
        lambda_hop_list = []
        
        
    total_experiment_hops = len(expdata)
    total_background_hops = len(background)
    
    
    
    for chrom in tqdm.tqdm(chrm):

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChrom.sort()
        
        max_pos = curChrom[-1] + 4
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        
        curbackgroundframe = np.array(list(background[background[0]==chrom][1]))
        
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        for window_start in range(0,int(max_pos+window_size),int(step_size)):

            num_exp_hops = _findinsertionslen(curChromnp, window_start, window_start+window_size - 1, length)
            if num_exp_hops > 1:
                num_bg_hops = _findinsertionslen(curbackgroundframe, window_start, window_start+window_size - 1, length)
                p = _compute_cumulative_poisson(num_exp_hops,num_bg_hops,total_experiment_hops,total_background_hops,pseudocounts)
            else:
                p = 1
                

            #is this window significant?
            if p < pvalue_cutoff:
                #was last window significant?
                if sig_flag:
                    #if so, extend end of windows
                    sig_end = window_start+window_size-1
                else:
                    #otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start+window_size-1
                    sig_flag = 1

            else:
                #current window not significant.  Was last window significant?
                if sig_flag:
                    
                    #add full sig window to the frame of peaks 
                    #add chr, peak start, peak end
                    chr_list.append(chrom) #add chr to frame
                    start_list.append(sig_start) #add peak start to frame
                    end_list.append(sig_end) #add peak end to frame
                    
                    #compute peak center and add to frame
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)

                    #add number of experiment hops in peak to frame
                    num_exp_hops = len(overlap)

                    #add number of background hops in peak to frame
                    num_bg_hops = _findinsertionslen(curbackgroundframe, sig_start, sig_end, length)
                    
                  
                    if record:
                        center_list.append(peak_center) #add peak center to frame
                        num_exp_hops_list.append(num_exp_hops)
                        #add fraction of experiment hops in peak to frame
                        frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                        tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)
                        num_bg_hops_list.append(num_bg_hops)
                        frac_bg_list.append(float(num_bg_hops)/total_background_hops)
                        tph_bg_list.append(float(num_bg_hops)*100000/total_background_hops)
                        

                    #find lambda and compute significance of peak
                    if total_background_hops >= total_experiment_hops: #scale bg hops down
                        #compute lambda bg
                        num_TTAAs = _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                        lambda_bg = ((num_bg_hops*(float(total_experiment_hops)/total_background_hops))/max(num_TTAAs,1)) 


                        #compute lambda 1k
                        num_bg_hops_1k = _findinsertionslen(curbackgroundframe, peak_center-499, peak_center+500, length)
                        num_TTAAs_1k = _findinsertionslen(curTTAAframe, peak_center-499, peak_center+500, length)
                        lambda_1k = (num_bg_hops_1k*(float(total_experiment_hops)/total_background_hops))/(max(num_TTAAs_1k,1))


                        #compute lambda 5k
                        num_bg_hops_5k = _findinsertionslen(curbackgroundframe, peak_center-2499, peak_center+2500, length)
                        num_TTAAs_5k = _findinsertionslen(curTTAAframe, peak_center-2499, peak_center+2500, length)
                        lambda_5k = (num_bg_hops_5k*(float(total_experiment_hops)/total_background_hops))/(max(num_TTAAs_5k,1))


                        #compute lambda 10k
                        num_bg_hops_10k = _findinsertionslen(curbackgroundframe, peak_center-4999, peak_center+5000, length)
                        num_TTAAs_10k = _findinsertionslen(curTTAAframe, peak_center-4999, peak_center+5000, length)
                        lambda_10k = (num_bg_hops_10k*(float(total_experiment_hops)/total_background_hops))/(max(num_TTAAs_10k,1))
                        lambda_f = max([lambda_bg,lambda_1k,lambda_5k,lambda_10k])


                        #record type of lambda used
                        index = [lambda_bg,lambda_1k,lambda_5k,lambda_10k].index(max([lambda_bg,lambda_1k,lambda_5k,lambda_10k]))
                        lambda_type_list.append(list_of_l_names[index])
                        #record lambda
                        lambda_list.append(lambda_f)
                        #compute pvalue and record it

                        pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),lambda_f*max(num_TTAAs,1)+pseudocounts)
                        pvalue_list.append(pvalue)


                        tph_bgs = float(num_exp_hops)*100000/total_experiment_hops-float(num_bg_hops)*100000/total_background_hops
                        
                        if record:
                            lambda_type_list.append(list_of_l_names[index])
                            lambda_list.append(lambda_f)
                            tph_bgs_list.append(tph_bgs)
                            lambda_hop_list.append(lambda_f*max(num_TTAAs,1))


                        index = [lambda_bg,lambda_1k,lambda_5k,lambda_10k].index(max([lambda_bg,lambda_1k,lambda_5k,lambda_10k]))
                        lambdatype = list_of_l_names[index]
                        #l = [pvalue,tph_bgs,lambda_f,lambdatype]

                    else: #scale experiment hops down
                        

                        #compute lambda bg
                        num_TTAAs = _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                        lambda_bg = (float(num_bg_hops)/max(num_TTAAs,1)) 


                        #compute lambda 1k
                        num_bg_hops_1k = _findinsertionslen(curbackgroundframe, peak_center-499, peak_center+500, length)
                        num_TTAAs_1k = _findinsertionslen(curTTAAframe, peak_center-499, peak_center+500, length)
                        lambda_1k = (float(num_bg_hops_1k)/(max(num_TTAAs_1k,1)))


                        #compute lambda 5k
                        num_bg_hops_5k = _findinsertionslen(curbackgroundframe, peak_center-2499, peak_center+2500, length)
                        num_TTAAs_5k = _findinsertionslen(curTTAAframe, peak_center-2499, peak_center+2500, length)
                        lambda_5k = (float(num_bg_hops_5k)/(max(num_TTAAs_5k,1)))


                        #compute lambda 10k
                        num_bg_hops_10k = _findinsertionslen(curbackgroundframe, peak_center-4999, peak_center+5000, length)
                        num_TTAAs_10k = _findinsertionslen(curTTAAframe, peak_center-4999, peak_center+5000, length)
                        lambda_10k = (float(num_bg_hops_10k)/(max(num_TTAAs_10k,1)))
                        lambda_f = max([lambda_bg,lambda_1k,lambda_5k,lambda_10k])


                        #record type of lambda used
                        index = [lambda_bg,lambda_1k,lambda_5k,lambda_10k].index(max([lambda_bg,
                                                                                      lambda_1k,lambda_5k,lambda_10k]))

                        #compute pvalue and record it
                        pvalue = 1-poisson.cdf(((float(total_background_hops)/total_experiment_hops)*num_exp_hops+ pseudocounts),lambda_f*max(num_TTAAs,1)+pseudocounts)
                        pvalue_list.append(pvalue)

                        tph_bgs = float(num_exp_hops)*100000/total_experiment_hops -float(num_bg_hops)*100000/total_background_hops

                        if record:
                            lambda_type_list.append(list_of_l_names[index])
                            lambda_list.append(lambda_f)
                            tph_bgs_list.append(tph_bgs)
                            lambda_hop_list.append(lambda_f*max(num_TTAAs,1))


                        index = [lambda_bg,lambda_1k,lambda_5k,lambda_10k].index(max([lambda_bg,
                                                                                      lambda_1k,lambda_5k,lambda_10k]))
                        lambdatype = list_of_l_names[index]
                        
                        


                    #number of hops that are a user-defined distance from peak center
                    sig_flag = 0
                        

    if record:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center","Experiment Hops",
                "Fraction Experiment","TPH Experiment","Lambda Type",
                "Lambda","Poisson pvalue"])


        peaks_frame["Lambda background hops"] = lambda_hop_list
        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Hops"] = num_exp_hops_list 
        peaks_frame["Fraction Experiment"] = frac_exp_list 
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Background Hops"] = num_bg_hops_list 
        peaks_frame["Fraction Background"] = frac_bg_list
        peaks_frame["TPH Background"] = tph_bg_list
        peaks_frame["TPH Background subtracted"] = tph_bgs_list
        peaks_frame["Lambda Type"] = lambda_type_list
        peaks_frame["Lambda"] = lambda_list
        
        
    else:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["Poisson pvalue"] = pvalue_list
        

    #peaks_frame = peaks_frame[peaks_frame["Poisson pvalue"] <= pvalue_cutoff]
    
    
    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]


def callpeaksMACS2_bfnew(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    min_hops: int = 3, 
    extend: int = 200,
    window_size: int = 1000, 
    step_size: int = 500, 
    pseudocounts: float = 0.2, 
    pvalue_cutoff: float = 0.01,
    lam_win_size: Optional[int] =1000000,
    record: bool = False, 
    test_method: _PeaktestMethod = "poisson") -> pd.DataFrame:
    
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
        
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    # create lists to record 
    chr_list = []
    start_list = []
    end_list = []
    
    if record:
        center_list = []
        num_exp_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        pvalue_list = []
        background_hops = []
        expect_hops = []
        
    total_experiment_hops = len(expdata)
    
    
    for chrom in tqdm.tqdm(chrm):
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        totalcurTTAA = len(curTTAAframe)
        if len(curTTAAframe) == 0:
            continue

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        
        max_pos = curChrom[-1]
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        totalcurChrom = len(curChromnp)
      


        # caluclate the ratio for TTAA and background 
        if lam_win_size == None:
            lambdacur = (totalcurChrom/totalcurTTAA) #expected ratio of hops per TTAA

        for window_start in range(curChrom[0],int(max_pos+2*window_size),step_size):

            num_exp_hops = _findinsertionslen(curChromnp, window_start, window_start+window_size - 1, length)

            if num_exp_hops >= min_hops:

                num_TTAAs_window = _findinsertionslen(curTTAAframe, window_start, window_start+window_size - 1, length)

                #is this window significant?
                if test_method == "poisson":

                    if lam_win_size == None:
                        pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),lambdacur*max(num_TTAAs_window,1)+pseudocounts)
                    else:
                        num_TTAA_hops_lambda = _findinsertionslen(curTTAAframe , window_start - int(lam_win_size/2) +1, 
                                                          window_start+window_size + int(lam_win_size/2) - 1, length)
                        num_exp_hops_lambda = _findinsertionslen(curChromnp , window_start - int(lam_win_size/2) +1, 
                                                          window_start+window_size + int(lam_win_size/2) - 1, length)
                        pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                     float(num_exp_hops_lambda/num_TTAA_hops_lambda)*max(num_TTAAs_window,1)+pseudocounts)

                elif test_method == "binomial":

                    if lam_win_size == None:
                        pvalue = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                       p=((num_TTAAs_window+pseudocounts)/totalcurTTAA) , alternative='greater').pvalue
                    else:
                        num_TTAA_hops_lambda = _findinsertionslen(curTTAAframe , window_start - int(lam_win_size/2) +1, 
                                                          window_start+window_size + int(lam_win_size/2) - 1, length)
                        num_exp_hops_lambda = _findinsertionslen(curChromnp , window_start - int(lam_win_size/2) +1, 
                                                          window_start+window_size + int(lam_win_size/2) - 1, length)
                        pvalue = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lambda, 
                                       p=((num_TTAAs_window+pseudocounts)/num_TTAA_hops_lambda) , alternative='greater').pvalue

            else:
                pvalue = 1

            if pvalue < pvalue_cutoff :

                #was last window significant?
                if sig_flag:

                    #if so, extend end of windows
                    sig_end = window_start+window_size-1

                else:

                    #otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start+window_size-1
                    sig_flag = 1

            else:

                #current window not significant.  Was last window significant?
                if sig_flag:

                    #compute peak center and add to frame
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)

                    # redefine the overlap
                    sig_start = overlap.min() - extend
                    sig_end = overlap.max() + length + extend
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    num_exp_hops = len(overlap)

                    num_TTAAs_window = _findinsertionslen(curTTAAframe, sig_start, sig_end, length)

                    #compute pvalue and record it  
                    if test_method == "poisson":

                        if lam_win_size == None:
                            pvalue_list.append(1-poisson.cdf((num_exp_hops+pseudocounts),lambdacur*max(num_TTAAs_window,1)+pseudocounts))
                        else:
                            num_exp_hops_lam_win_size = _findinsertionslen(curChromnp, peak_center-(lam_win_size/2-1), 
                                                                          peak_center+(lam_win_size/2), length)
                            num_TTAAs_lam_win_size = _findinsertionslen(curTTAAframe, peak_center-(lam_win_size/2-1), 
                                                                       peak_center+(lam_win_size/2), length)
                            lambda_lam_win_size = float(num_exp_hops_lam_win_size)/(max(num_TTAAs_lam_win_size,1))
                            pvalue_list.append(1-poisson.cdf((num_exp_hops+pseudocounts), lambda_lam_win_size*max(num_TTAAs_window,1)+pseudocounts))

                    elif test_method == "binomial":

                        if lam_win_size == None:
                            pvalue_list.append(binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                           p=((num_TTAAs_window+pseudocounts)/totalcurTTAA) , alternative='greater').pvalue)
                        else:
                            num_exp_hops_lam_win_size = _findinsertionslen(curChromnp, peak_center-(lam_win_size/2-1), 
                                                                          peak_center+(lam_win_size/2), length)
                            num_TTAAs_lam_win_size = _findinsertionslen(curTTAAframe, peak_center-(lam_win_size/2-1), 
                                                                       peak_center+(lam_win_size/2), length)

                            pvalue_list.append(binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam_win_size, 
                                           p=((num_TTAAs_window+pseudocounts)/num_TTAAs_lam_win_size) , alternative='greater').pvalue)

                    if record:

                        # add chr, peak start, peak end
                        chr_list.append(chrom) #add chr to frame
                        start_list.append(sig_start) #add peak start to frame
                        end_list.append(sig_end) #add peak end to frame
                        center_list.append(peak_center) #add peak center to frame
                        num_exp_hops_list.append(num_exp_hops)

                        #add fraction of experiment hops in peak to frame
                        frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                        tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)

                        background_hops.append(num_TTAAs_window)

                        if lam_win_size == None:
                            expect_hops.append(lambdacur*max(num_TTAAs_window,1)+pseudocounts)
                        else:
                            expect_hops.append(float(num_exp_hops_lam_win_size)/(max(num_TTAAs_lam_win_size,1))*max(num_TTAAs_window,1)+pseudocounts)

                    sig_flag = 0



    if record:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center","pvalue","Experiment Hops","TTAA Hops",
                "Fraction Experiment","TPH Experiment"])

        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Hops"] = num_exp_hops_list 
        peaks_frame["Fraction Experiment"] = frac_exp_list 
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["TTAA Hops"] = background_hops
        peaks_frame["Expect Hops"] = expect_hops

    else:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","pvalue"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["pvalue"] = pvalue_list

    peaks_frame = peaks_frame[peaks_frame["pvalue"] <= pvalue_cutoff]

    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]

def callpeaksMACS2_bfnew2(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    min_hops: int = 3, 
    extend: int = 200,
    window_size: int = 1000, 
    step_size: int = 500, 
    pseudocounts: float = 0.2, 
    pvalue_cutoff: float = 0.01,
    lam_win_size: Optional[int] = None,
    record: bool = False, 
    test_method: _PeaktestMethod = "poisson") -> pd.DataFrame:
    
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
        
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    # create lists to record 
    chr_list = []
    start_list = []
    end_list = []
    
    if record:
        center_list = []
        num_exp_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        pvalue_list = []
        background_hops = []
        expect_hops = []
        
    total_experiment_hops = len(expdata)
    
    
    for chrom in tqdm.tqdm(chrm):
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        totalcurTTAA = len(curTTAAframe)
        if len(curTTAAframe) == 0:
            continue

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        
        max_pos = curChrom[-1]
        sig_start = 0
        sig_end = 0
        sig_flag = 0

        
        totalcurChrom = len(curChromnp)
        
        
        starthops1 = 0
        startTTAA1 = 0
      

        startTTAA2 = 0
        
        if lam_win_size != None:
            
            starthopslam1 = 0
            startTTAAlam1 = 0

            starthopslam2 = 0
            startTTAAlam2 = 0
            
        if totalcurTTAA != 0:

            # caluclate the ratio for TTAA and background 
            if lam_win_size == None:
                lambdacur = (totalcurChrom/totalcurTTAA) #expected ratio of hops per TTAA

            for window_start in range(curChrom[0],int(max_pos+2*window_size),step_size):

                num_exp_hops, starthops1 = _findinsertionslen2(curChromnp, window_start, window_start+window_size - 1, 
                                                              length, starthops1, totalcurChrom)
                
                if num_exp_hops >= min_hops:
                    
                    num_TTAAs_window, startTTAA1 = _findinsertionslen2(curTTAAframe, window_start, window_start+window_size - 1, 
                                                                     length, startTTAA1, totalcurTTAA)
                    
                    #is this window significant?
                    if test_method == "poisson":
                        
                        if lam_win_size == None:
                            pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),lambdacur*max(num_TTAAs_window,1)+pseudocounts)
                        else:
                            num_TTAA_hops_lambda, startTTAAlam1 = _findinsertionslen2(curTTAAframe , 
                                                                                     window_start - int(lam_win_size/2) +1, 
                                                                                     window_start+window_size + int(lam_win_size/2) - 1, 
                                                                                     length, startTTAAlam1, totalcurTTAA)
                           
                            num_exp_hops_lambda, starthopslam1 = _findinsertionslen2(curChromnp , 
                                                                                    window_start - int(lam_win_size/2) +1, 
                                                                                     window_start+window_size + int(lam_win_size/2) - 1, 
                                                                                    length, starthopslam1, totalcurChrom)
                    
                            
                            pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                         float(num_exp_hops_lambda/num_TTAA_hops_lambda)*max(num_TTAAs_window,1)+pseudocounts)
                            
                    elif test_method == "binomial":
                        
                        if lam_win_size == None:
                            pvalue = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                           p=((num_TTAAs_window+pseudocounts)/totalcurTTAA) , alternative='greater').pvalue
                        else:
                            num_TTAA_hops_lambda, startTTAAlam1 = _findinsertionslen2(curTTAAframe , 
                                                                                     window_start - int(lam_win_size/2) +1, 
                                                                                     window_start+window_size + int(lam_win_size/2) - 1, 
                                                                                     length, startTTAAlam1, totalcurTTAA)
                            num_exp_hops_lambda, starthopslam1 = _findinsertionslen2(curChromnp , 
                                                                                    window_start - int(lam_win_size/2) +1, 
                                                                                     window_start+window_size + int(lam_win_size/2) - 1, 
                                                                                    length, starthopslam1, totalcurChrom)
                            pvalue = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lambda, 
                                           p=((num_TTAAs_window+pseudocounts)/num_TTAA_hops_lambda) , alternative='greater').pvalue

                else:
                    pvalue = 1

                if pvalue < pvalue_cutoff :
                    
                    #was last window significant?
                    if sig_flag:
                        
                        #if so, extend end of windows
                        sig_end = window_start+window_size-1
                        
                    else:
                        
                        #otherwise, define new start and end and set flag
                        sig_start = window_start
                        sig_end = window_start+window_size-1
                        sig_flag = 1

                else:
                    
                    #current window not significant.  Was last window significant?
                    if sig_flag:

                        #compute peak center and add to frame
                        
                        overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                        peak_center = np.median(overlap)
                        
                        # redefine the overlap
                        sig_start = overlap.min() - extend
                        sig_end = overlap.max() + length + extend
                        overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                        num_exp_hops = len(overlap)
                

                        num_TTAAs_window, startTTAA2 = _findinsertionslen2(curTTAAframe, sig_start, sig_end, length, 
                                                                          startTTAA2, totalcurTTAA)
                        num_TTAAs_window2 = _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                      
                        
                        #num_TTAAs_window= _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                        #                                                  startTTAA2, totalcurTTAA)

                        #compute pvalue and record it  
                        if test_method == "poisson":
                            
                            if lam_win_size == None:
                                pvalue_list.append(1-poisson.cdf((num_exp_hops+pseudocounts),lambdacur*max(num_TTAAs_window,1)+pseudocounts))
                            else:
                                num_exp_hops_lam_win_size, starthopslam2  = _findinsertionslen2(curChromnp, peak_center-(lam_win_size/2-1), 
                                                                              peak_center+(lam_win_size/2), length, 
                                                                                starthopslam2 ,totalcurChrom)
                         
                                
                                num_TTAAs_lam_win_size , startTTAAlam2= _findinsertionslen2(curTTAAframe, peak_center-(lam_win_size/2-1), 
                                                                           peak_center+(lam_win_size/2), length, startTTAAlam2,
                                                                            totalcurTTAA)
                                
                           
                                lambda_lam_win_size = float(num_exp_hops_lam_win_size)/(max(num_TTAAs_lam_win_size,1))
                                pvalue_list.append(1-poisson.cdf((num_exp_hops+pseudocounts), lambda_lam_win_size*max(num_TTAAs_window,1)+pseudocounts))
                            
                        elif test_method == "binomial":
                            
                            if lam_win_size == None:
                                pvalue_list.append(binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                               p=((num_TTAAs_window+pseudocounts)/totalcurTTAA) , alternative='greater').pvalue)
                            else:
                                num_exp_hops_lam_win_size, starthopslam2  = _findinsertionslen2(curChromnp, peak_center-(lam_win_size/2-1), 
                                                                              peak_center+(lam_win_size/2), length, 
                                                                                starthopslam2 ,totalcurChrom)
                            
                                
                            
                                num_TTAAs_lam_win_size , startTTAAlam2= _findinsertionslen2(curTTAAframe, peak_center-(lam_win_size/2-1), 
                                                                           peak_center+(lam_win_size/2), length, startTTAAlam2,
                                                                            totalcurTTAA)
                                pvalue_list.append(binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam_win_size, 
                                               p=((num_TTAAs_window+pseudocounts)/num_TTAAs_lam_win_size) , alternative='greater').pvalue)

                        if record:

                            # add chr, peak start, peak end
                            chr_list.append(chrom) #add chr to frame
                            start_list.append(sig_start) #add peak start to frame
                            end_list.append(sig_end) #add peak end to frame
                            center_list.append(peak_center) #add peak center to frame
                            num_exp_hops_list.append(num_exp_hops)

                            #add fraction of experiment hops in peak to frame
                            frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                            tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)
                      
                            background_hops.append(num_TTAAs_window)
                        
                            if lam_win_size == None:
                                expect_hops.append(lambdacur*max(num_TTAAs_window,1)+pseudocounts)
                            else:
                                expect_hops.append(float(num_exp_hops_lam_win_size)/(max(num_TTAAs_lam_win_size,1))*max(num_TTAAs_window,1)+pseudocounts)

                        sig_flag = 0



        if record:
            peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center","pvalue","Experiment Hops","TTAA Hops",
                    "Fraction Experiment","TPH Experiment"])

            peaks_frame["Center"] = center_list
            peaks_frame["Experiment Hops"] = num_exp_hops_list 
            peaks_frame["Fraction Experiment"] = frac_exp_list 
            peaks_frame["TPH Experiment"] = tph_exp_list
            peaks_frame["TTAA Hops"] = background_hops
            peaks_frame["Expect Hops"] = expect_hops

        else:
            peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","pvalue"])

        peaks_frame["Chr"] = chr_list
        peaks_frame["Start"] = start_list
        peaks_frame["End"] = end_list
        peaks_frame["pvalue"] = pvalue_list

        peaks_frame = peaks_frame[peaks_frame["pvalue"] <= pvalue_cutoff]

    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]


    
def callpeaksMACS2new(
    expdata: pd.DataFrame, 
    background: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    extend: int = 200, 
    lam_win_size: Optional[int] = 100000,
    pvalue_cutoff_background: float = 0.01,  
    pvalue_cutoff_TTAA: float = 0.01,
    window_size: int = 1000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    test_method: _PeaktestMethod = "poisson",
    min_hops: int = 3,
    record: bool = False)-> pd.DataFrame:
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    # create lists to record the basic information
    chr_list = []
    start_list = []
    end_list = []
    pvalue_list_background = []
    pvalue_list_TTAA = []
    
    if record:
        # create lists to record other information
        center_list = []
        num_exp_hops_list = []
        num_bg_hops_list = []
        num_TTAA_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        frac_bg_list = []
        tph_bg_list = []
        tph_bgs_list = []
        
    # record total number of hops  
    total_experiment_hops = len(expdata)
    total_background_hops = len(background)
    
    # going from the first Chromosome to the last
    for chrom in tqdm.tqdm(chrm):

        curbackgroundframe = np.array(list(background[background[0]==chrom][1]))
        totalcurbackground = len(curbackgroundframe)
        if totalcurbackground == 0:
            continue

        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        totalcurTTAA = len(curTTAAframe)
        if totalcurTTAA == 0:
            continue
        
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        

        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        
        # initial the parameters
        max_pos = curChrom[-1] + length +1
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        # calculate the total number of hops
        totalcurChrom = len(curChromnp)
        


        if lam_win_size == None:
        # caluclate the ratio for TTAA and background 
            lambdacurTTAA = float(totalcurChrom/totalcurTTAA) #expected ratio of hops per TTAA
            lambdacurbackground = float(totalcurChrom/totalcurbackground) #expected ratio of hops per background


        for window_start in range(curChromnp[0],int(max_pos+2*window_size),int(step_size)):

            num_exp_hops = _findinsertionslen(curChromnp, window_start, window_start+window_size - 1, length)

            if num_exp_hops >= min_hops :

                # find out the number of hops in the current window for backgound 
                num_bg_hops = _findinsertionslen(curbackgroundframe, window_start, window_start+window_size - 1, length)


                if num_bg_hops >0 :

                    if lam_win_size == None:

                        if test_method == "poisson":
                            pvaluebg = _compute_cumulative_poisson(num_exp_hops,
                                                                  num_bg_hops,totalcurChrom,
                                                                  totalcurbackground,pseudocounts)
                        elif test_method == "binomial":
                            pvaluebg = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                               p=((num_bg_hops+pseudocounts)/totalcurbackground) , 
                                                 alternative='greater').pvalue
                    else:

                        num_exp_hops_lam = _findinsertionslen(curChromnp, window_start - int(lam_win_size/2) +1,
                                                window_start+window_size + int(lam_win_size/2) - 1, length)
                        num_exp_bg_lam = _findinsertionslen(curbackgroundframe, 
                                                             window_start - int(lam_win_size/2) +1,
                                                             window_start+window_size + int(lam_win_size/2) - 1,
                                                             length)

                        if test_method == "poisson":
                            pvaluebg = _compute_cumulative_poisson(num_exp_hops,
                                                                  num_bg_hops,num_exp_hops_lam,
                                                                  num_exp_bg_lam,pseudocounts)
                        elif test_method == "binomial":
                            pvaluebg = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam, 
                                               p=((num_bg_hops+pseudocounts)/num_exp_bg_lam) , 
                                                 alternative='greater').pvalue


                else:
                    pvaluebg = 0

                # if it passes, then look at the TTAA:
                if pvaluebg < pvalue_cutoff_background :

                    num_TTAA_hops = _findinsertionslen(curTTAAframe, window_start, 
                                                      window_start+window_size - 1, length)

                    if lam_win_size == None:

                        if test_method == "poisson":
                            pvalueTTAA = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                                       lambdacurTTAA*num_TTAA_hops+pseudocounts)
                        elif test_method == "binomial":
                            pvalueTTAA = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                               p=((num_TTAA_hops+pseudocounts)/totalcurTTAA) , 
                                                   alternative='greater').pvalue
                    else:


                        num_TTAA_hops_lam = _findinsertionslen(curTTAAframe, 
                                                             window_start - int(lam_win_size/2) +1,
                                                             window_start+window_size + int(lam_win_size/2) - 1,
                                                             length)
                        if test_method == "poisson":
                            pvalueTTAA = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                                       (num_exp_hops_lam/num_TTAA_hops_lam)*num_TTAA_hops+
                                                       pseudocounts)
                        elif test_method == "binomial":
                            pvalueTTAA = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam, 
                                               p=((num_TTAA_hops+pseudocounts)/num_TTAA_hops_lam) , 
                                                   alternative='greater').pvalue

                else:
                    pvaluebg = 1
                    pvalueTTAA = 1


            else:
                pvaluebg = 1
                pvalueTTAA = 1


            #is this window significant?
            if pvaluebg < pvalue_cutoff_background and pvalueTTAA < pvalue_cutoff_TTAA :
                #was last window significant?
                if sig_flag:
                    #if so, extend end of windows
                    sig_end = window_start+window_size-1
                else:
                    #otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start+window_size-1
                    sig_flag = 1

            else:
                #current window not significant.  Was last window significant?
                if sig_flag:

                    # Let's first give a initial view of our peak
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)

                    # redefine the overlap
                    sig_start = overlap.min() - extend
                    sig_end = overlap.max() + 3 + extend

                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    num_exp_hops = len(overlap)

                    #add number of background hops in peak to frame
                    num_TTAA_hops = _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                    num_bg_hops = _findinsertionslen(curbackgroundframe, sig_start, sig_end, length)


                    if record:
                        chr_list.append(chrom) #add chr to frame
                        start_list.append(sig_start) #add peak start to frame
                        end_list.append(sig_end) #add peak end to frame
                        center_list.append(peak_center) #add peak center to frame
                        num_TTAA_hops_list.append(num_TTAA_hops)
                        num_exp_hops_list.append(num_exp_hops)#add fraction of experiment hops in peak to frame
                        frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                        tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)
                        num_bg_hops_list.append(num_bg_hops)
                        frac_bg_list.append(float(num_bg_hops)/total_background_hops)
                        tph_bg_list.append(float(num_bg_hops)*100000/total_background_hops)
                        tph_bgs = float(num_exp_hops)*100000/total_experiment_hops-float(num_bg_hops)*100000/total_background_hops
                        tph_bgs_list.append(tph_bgs)

                    # caluclate the final P value 

                    if lam_win_size == None:

                        if test_method == "poisson":

                            pvalue_list_TTAA.append(1-poisson.cdf((num_exp_hops+pseudocounts),
                                                                  lambdacurTTAA*num_TTAA_hops+pseudocounts))
                            pvalue_list_background.append(_compute_cumulative_poisson(num_exp_hops,
                                                                                     num_bg_hops,
                                                                                     totalcurChrom,
                                                                                     totalcurbackground,
                                                                                     pseudocounts))


                        elif test_method == "binomial":

                            pvalue_list_TTAA.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                              n=totalcurChrom, 
                                                              p=((num_TTAA_hops+pseudocounts)/totalcurTTAA) , 
                                                              alternative='greater').pvalue)
                            pvalue_list_background.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                                n=totalcurChrom, 
                                                                p=((num_bg_hops+pseudocounts)/totalcurbackground) , 
                                                                alternative='greater').pvalue)
                    else:

                        num_exp_hops_lam = _findinsertionslen(curChromnp, sig_start - int(lam_win_size/2) +1,
                                                             sig_end + int(lam_win_size/2) - 1, length)

                        num_exp_bg_lam = _findinsertionslen(curbackgroundframe, 
                                                             sig_start - int(lam_win_size/2) +1,
                                                             sig_end + int(lam_win_size/2) - 1,
                                                             length)

                        num_exp_TTAA_lam = _findinsertionslen(curTTAAframe, 
                                                             sig_start - int(lam_win_size/2) +1,
                                                             sig_end + int(lam_win_size/2) - 1,
                                                             length)

                        if test_method == "poisson":

                            pvalue_list_TTAA.append(1-poisson.cdf((num_exp_hops+pseudocounts),
                                                                  (num_exp_hops_lam/num_exp_TTAA_lam)*num_TTAA_hops
                                                                  +pseudocounts))
                            pvalue_list_background.append(_compute_cumulative_poisson(num_exp_hops,
                                                                                     num_bg_hops,
                                                                                     num_exp_hops_lam,
                                                                                     num_exp_bg_lam,
                                                                                     pseudocounts))


                        elif test_method == "binomial":

                            pvalue_list_TTAA.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                              n=num_exp_hops_lam, 
                                                              p=((num_TTAA_hops+pseudocounts)/num_exp_TTAA_lam) , 
                                                              alternative='greater').pvalue)
                            pvalue_list_background.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                                n=num_exp_hops_lam, 
                                                                p=((num_bg_hops+pseudocounts)/num_exp_bg_lam) , 
                                                                alternative='greater').pvalue)



                    #number of hops that are a user-defined distance from peak center
                    sig_flag = 0


    if record:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center",
                                              "Experiment Hops","Background Hops","TTAA Hops",
                                              "pvalue TTAA","pvalue background"])


        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Hops"] = num_exp_hops_list 
        peaks_frame["Background Hops"] = num_bg_hops_list 
        peaks_frame["TTAA Hops"] = num_TTAA_hops_list

        peaks_frame["Fraction Experiment"] = frac_exp_list 
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Fraction Background"] = frac_bg_list
        peaks_frame["TPH Background"] = tph_bg_list
        peaks_frame["TPH Background subtracted"] = tph_bgs_list


    else:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["pvalue TTAA"] = pvalue_list_TTAA
    peaks_frame["pvalue background"] = pvalue_list_background


    peaks_frame = peaks_frame[peaks_frame["pvalue TTAA"] <= pvalue_cutoff_TTAA]
    peaks_frame = peaks_frame[peaks_frame["pvalue background"] <= pvalue_cutoff_background]


    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]


    
def callpeaksMACS2new2(
    expdata: pd.DataFrame, 
    background: pd.DataFrame,  
    TTAAframe: pd.DataFrame, 
    length: int, 
    extend: int = 200, 
    lam_win_size: Optional[int] = 100000,
    pvalue_cutoff_background: float = 0.01,  
    pvalue_cutoff_TTAA: float = 0.01,
    window_size: int = 1000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    test_method: _PeaktestMethod = "poisson",
    min_hops: int = 3,
    record: bool = False) -> pd.DataFrame:
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    # create lists to record the basic information
    chr_list = []
    start_list = []
    end_list = []
    pvalue_list_background = []
    pvalue_list_TTAA = []
    
    if record:
        # create lists to record other information
        center_list = []
        num_exp_hops_list = []
        num_bg_hops_list = []
        num_TTAA_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        frac_bg_list = []
        tph_bg_list = []
        tph_bgs_list = []
        
    # record total number of hops  
    total_experiment_hops = len(expdata)
    total_background_hops = len(background)
    
    # going from the first Chromosome to the last
    for chrom in tqdm.tqdm(chrm):

        curbackgroundframe = np.array(list(background[background[0]==chrom][1]))
        totalcurbackground = len(curbackgroundframe)
        if totalcurbackground == 0:
            continue

        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        totalcurTTAA = len(curTTAAframe)
        if totalcurTTAA == 0:
            continue
        
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        curbackgroundframe.sort()
        
        # initial the parameters
        max_pos = curChrom[-1] + length +1
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        
        
        # calculate the total number of hops
        totalcurChrom = len(curChromnp)
        
        starthop1 = 0
        startTTAA1 = 0
        startbg1 = 0
        
        starthop2 = 0
        startTTAA2 = 0
        startbg2 = 0
   

        if lam_win_size == None:
        
        # caluclate the ratio for TTAA and background 
            lambdacurTTAA = float(totalcurChrom/totalcurTTAA) #expected ratio of hops per TTAA
            lambdacurbackground = float(totalcurChrom/totalcurbackground) #expected ratio of hops per background
            
        else:
            
            startTTAAlam1 = 0
            startbglam1 = 0
            starthoplam = 0
            
            startTTAAlam2 = 0
            startbglam2 = 0
            starthoplam2 = 0

        for window_start in range(curChromnp[0],int(max_pos+2*window_size),int(step_size)):

            num_exp_hops, starthop1 = _findinsertionslen2(curChromnp, window_start, window_start+window_size - 1, 
                                                        length, starthop1, totalcurChrom)
            

            if num_exp_hops >= min_hops :

                # find out the number of hops in the current window for backgound 
                num_bg_hops, startbg1 = _findinsertionslen2(curbackgroundframe, window_start, 
                                                           window_start+window_size - 1,length,
                                                           startbg1, totalcurbackground)


                if num_bg_hops >0 :

                    if lam_win_size == None:

                        if test_method == "poisson":
                            pvaluebg = _compute_cumulative_poisson(num_exp_hops,
                                                                  num_bg_hops,totalcurChrom,
                                                                  totalcurbackground,pseudocounts)
                        elif test_method == "binomial":
                            pvaluebg = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                               p=((num_bg_hops+pseudocounts)/totalcurbackground) , 
                                                 alternative='greater').pvalue
                    else:

                        num_exp_hops_lam, starthoplam =_findinsertionslen2(curChromnp, 
                                                                window_start - int(lam_win_size/2) +1,
                                                                    window_start+window_size + int(lam_win_size/2) - 1, 
                                                                           length,starthoplam, totalcurChrom)
                        
                        num_exp_bg_lam, startbglam1 = _findinsertionslen2(curbackgroundframe, 
                                                             window_start - int(lam_win_size/2) +1,
                                                             window_start+window_size + int(lam_win_size/2) - 1,
                                                             length,startbglam1, totalcurbackground)

                        if test_method == "poisson":
                            pvaluebg = _compute_cumulative_poisson(num_exp_hops,
                                                                  num_bg_hops,num_exp_hops_lam,
                                                                  num_exp_bg_lam,pseudocounts)
                        elif test_method == "binomial":
                            pvaluebg = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam, 
                                               p=((num_bg_hops+pseudocounts)/num_exp_bg_lam) , 
                                                 alternative='greater').pvalue


                else:
                    pvaluebg = 0

                # if it passes, then look at the TTAA:
                if pvaluebg < pvalue_cutoff_background :

                    num_TTAA_hops, startTTAA1 = _findinsertionslen2(curTTAAframe, window_start, 
                                                      window_start+window_size - 1, length,
                                                      startTTAA1, totalcurTTAA)

                    if lam_win_size == None:

                        if test_method == "poisson":
                            pvalueTTAA = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                                       lambdacurTTAA*num_TTAA_hops+pseudocounts)
                        elif test_method == "binomial":
                            pvalueTTAA = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                               p=((num_TTAA_hops+pseudocounts)/totalcurTTAA) , 
                                                   alternative='greater').pvalue
                    else:


                        num_TTAA_hops_lam, startTTAAlam1 = _findinsertionslen2(curTTAAframe, 
                                                             window_start - int(lam_win_size/2) +1,
                                                             window_start+window_size + int(lam_win_size/2) - 1,
                                                             length, startTTAAlam1, totalcurTTAA)
                        if test_method == "poisson":
                            pvalueTTAA = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                                       (num_exp_hops_lam/num_TTAA_hops_lam)*num_TTAA_hops+
                                                       pseudocounts)
                        elif test_method == "binomial":
                            pvalueTTAA = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam, 
                                               p=((num_TTAA_hops+pseudocounts)/num_TTAA_hops_lam) , 
                                                   alternative='greater').pvalue

                else:
                    pvaluebg = 1
                    pvalueTTAA = 1


            else:
                pvaluebg = 1
                pvalueTTAA = 1


            #is this window significant?
            if pvaluebg < pvalue_cutoff_background and pvalueTTAA < pvalue_cutoff_TTAA :
                #was last window significant?
                if sig_flag:
                    #if so, extend end of windows
                    sig_end = window_start+window_size-1
                else:
                    #otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start+window_size-1
                    sig_flag = 1

            else:
                #current window not significant.  Was last window significant?
                if sig_flag:

                    # Let's first give a initial view of our peak
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)

                    # redefine the overlap
                    sig_start = overlap.min() - extend
                    sig_end = overlap.max() + 3 + extend

                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    num_exp_hops = len(overlap)

                    #add number of background hops in peak to frame
                    num_TTAA_hops, startTTAA2 = _findinsertionslen2(curTTAAframe, sig_start, sig_end, 
                                                                      length, startTTAA2, totalcurTTAA)
                    num_bg_hops, startbg2 = _findinsertionslen2(curbackgroundframe, sig_start, sig_end, 
                                                     length, startbg2, totalcurbackground)


                    if record:
                        chr_list.append(chrom) #add chr to frame
                        start_list.append(sig_start) #add peak start to frame
                        end_list.append(sig_end) #add peak end to frame
                        center_list.append(peak_center) #add peak center to frame
                        num_TTAA_hops_list.append(num_TTAA_hops)
                        num_exp_hops_list.append(num_exp_hops)#add fraction of experiment hops in peak to frame
                        frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                        tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)
                        num_bg_hops_list.append(num_bg_hops)
                        frac_bg_list.append(float(num_bg_hops)/total_background_hops)
                        tph_bg_list.append(float(num_bg_hops)*100000/total_background_hops)
                        tph_bgs = float(num_exp_hops)*100000/total_experiment_hops-float(num_bg_hops)*100000/total_background_hops
                        tph_bgs_list.append(tph_bgs)

                    # caluclate the final P value 

                    if lam_win_size == None:

                        if test_method == "poisson":

                            pvalue_list_TTAA.append(1-poisson.cdf((num_exp_hops+pseudocounts),
                                                                  lambdacurTTAA*num_TTAA_hops+pseudocounts))
                            pvalue_list_background.append(_compute_cumulative_poisson(num_exp_hops,
                                                                                     num_bg_hops,
                                                                                     totalcurChrom,
                                                                                     totalcurbackground,
                                                                                     pseudocounts))


                        elif test_method == "binomial":

                            pvalue_list_TTAA.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                              n=totalcurChrom, 
                                                              p=((num_TTAA_hops+pseudocounts)/totalcurTTAA) , 
                                                              alternative='greater').pvalue)
                            pvalue_list_background.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                                n=totalcurChrom, 
                                                                p=((num_bg_hops+pseudocounts)/totalcurbackground) , 
                                                                alternative='greater').pvalue)
                    else:

                        num_exp_hops_lam , starthoplam2= _findinsertionslen2(curChromnp, 
                                                                           sig_start - int(lam_win_size/2) +1,
                                                                            sig_end + int(lam_win_size/2) - 1, 
                                                                            length, starthoplam2, totalcurChrom)

                        num_exp_bg_lam, startbglam2 = _findinsertionslen2(curbackgroundframe, 
                                                             sig_start - int(lam_win_size/2) +1,
                                                             sig_end + int(lam_win_size/2) - 1,
                                                             length,startbglam2, totalcurbackground)

                        num_exp_TTAA_lam, startTTAAlam2 = _findinsertionslen2(curTTAAframe, 
                                                             sig_start - int(lam_win_size/2) +1,
                                                             sig_end + int(lam_win_size/2) - 1,
                                                             length, startTTAAlam2, totalcurTTAA)

                        if test_method == "poisson":

                            pvalue_list_TTAA.append(1-poisson.cdf((num_exp_hops+pseudocounts),
                                                                  (num_exp_hops_lam/num_exp_TTAA_lam)*num_TTAA_hops
                                                                  +pseudocounts))
                            pvalue_list_background.append(_compute_cumulative_poisson(num_exp_hops,
                                                                                     num_bg_hops,
                                                                                     num_exp_hops_lam,
                                                                                     num_exp_bg_lam,
                                                                                     pseudocounts))


                        elif test_method == "binomial":

                            pvalue_list_TTAA.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                              n=num_exp_hops_lam, 
                                                              p=((num_TTAA_hops+pseudocounts)/num_exp_TTAA_lam) , 
                                                              alternative='greater').pvalue)
                            pvalue_list_background.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                                n=num_exp_hops_lam, 
                                                                p=((num_bg_hops+pseudocounts)/num_exp_bg_lam) , 
                                                                alternative='greater').pvalue)



                    #number of hops that are a user-defined distance from peak center
                    sig_flag = 0


    if record:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center",
                                              "Experiment Hops","Background Hops","TTAA Hops",
                                              "pvalue TTAA","pvalue background"])


        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Hops"] = num_exp_hops_list 
        peaks_frame["Background Hops"] = num_bg_hops_list 
        peaks_frame["TTAA Hops"] = num_TTAA_hops_list

        peaks_frame["Fraction Experiment"] = frac_exp_list 
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Fraction Background"] = frac_bg_list
        peaks_frame["TPH Background"] = tph_bg_list
        peaks_frame["TPH Background subtracted"] = tph_bgs_list


    else:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["pvalue TTAA"] = pvalue_list_TTAA
    peaks_frame["pvalue background"] = pvalue_list_background


    peaks_frame = peaks_frame[peaks_frame["pvalue TTAA"] <= pvalue_cutoff_TTAA]
    peaks_frame = peaks_frame[peaks_frame["pvalue background"] <= pvalue_cutoff_background]


    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]
    
    
def _checkint(number,name):
    
    try:
        number = int(number)
    except:
        print('Please enter a valid positive number or 0 for' + name)
    if number <0 :
        raise ValueError('Please enter a valid positive number or 0 for' + name)
       
    return number

def _checkpvalue(number,name):
    
    try:
        number = float(number)
    except:
        print('Please enter a valid number (0,1) for ' + name)
    if number <0 or number >1 :
        raise ValueError('Please enter a valid number (0,1) for ' + name)
       
    return number

def _check_test_method(method):
    if method != "poisson" and  method != "binomial" :
        raise ValueError("Not valid a valid test method. Please input poisson or binomial.")

    
_Peakcalling_Method = Optional[Literal["test","MACS2","Blockify"]]
_reference = Optional[Literal["hg38","mm10","yeast"]]

def callpeaks(
    expdata: pd.DataFrame, 
    background: Optional[pd.DataFrame] = None, 
    method: _Peakcalling_Method = "test", 
    reference: _reference = "hg38",
    pvalue_cutoff: float = 0.0001,  
    pvalue_cutoffbg: float = 0.0001, 
    pvalue_cutoffTTAA: float = 0.00001,
    min_hops: int = 5, 
    minlen: int = 0, 
    extend: int = 200, 
    maxbetween: int = 1500, 
    test_method: _PeaktestMethod = "poisson",
    window_size: int = 1000, 
    lam_win_size: Optional[int]  =100000, 
    step_size: int = 500, 
    pseudocounts: float = 0.2, 
    record: bool = True, 
    save: Optional[str] = None
    ) -> pd.DataFrame:
    
    # function for calling peaks
    
    # first make sure everything is correct:
    # check expdata
    if type(expdata) != pd.DataFrame :
        raise ValueError("Please input a pandas dataframe as the expression data.")
        
    if type(record) != bool:
                raise ValueError('Please enter a True/ False for record')
      
    if type(background) == pd.DataFrame :
        
        length = 3
        
        if method == "MACS2":
            
            print("For the MACS2 method with background, [expdata, background, reference, pvalue_cutoffbg, pvalue_cutoffTTAA, lam_win_size, window_size, step_size, extend, pseudocounts, test_method, min_hops, record] would be utilized.")
            
            if reference == "hg38":
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_hg38_ccf_new.bed",delimiter="\t",header=None)
            elif reference == "mm10":
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_mm10_ccf_new.bed",delimiter="\t",header=None)
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoffbg,"pvalue_cutoffbg")
            _checkpvalue(pvalue_cutoffTTAA,"pvalue_cutoffTTAA")
            
            window_size = _checkint(window_size,"window_size")
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size,"lam_win_size")
            extend = _checkint(extend,"extend")
            step_size = _checkint(step_size,"step_size")
            
                
            _check_test_method(test_method)
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
        
            if save == None:
                
                return callpeaksMACS2new2(expdata, background, TTAAframe, length, extend = extend, lam_win_size = lam_win_size,
                      pvalue_cutoff_background =  pvalue_cutoffbg,  pvalue_cutoff_TTAA = pvalue_cutoffTTAA,
                      window_size = window_size, step_size = step_size, pseudocounts = pseudocounts,
                      test_method= test_method, min_hops = min_hops, record = record)
            else:
                
                data = callpeaksMACS2new2(expdata, background, TTAAframe, length, extend = extend, lam_win_size = lam_win_size,
                      pvalue_cutoff_background =  pvalue_cutoffbg,  pvalue_cutoff_TTAA = pvalue_cutoffTTAA,
                      window_size = window_size, step_size = step_size, pseudocounts = pseudocounts,
                      test_method= test_method, min_hops = min_hops, record = record)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
                
        
        elif method == "test":
            
            print("For the test method with background, [expdata, background, reference, pvalue_cutoffbg, pvalue_cutoffTTAA, lam_win_size, pseudocounts, minlen, extend, maxbetween, test_method, min_hops, record] would be utilized.")
            
            if reference == "hg38":
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_hg38_ccf_new.bed",delimiter="\t",header=None)
            elif reference == "mm10":
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_mm10_ccf_new.bed",delimiter="\t",header=None)
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoffbg,"pvalue_cutoffbg")
            _checkpvalue(pvalue_cutoffTTAA,"pvalue_cutoffTTAA")
                
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size,"lam_win_size")
            extend = _checkint(extend,"extend")
            _checkint(pseudocounts,"pseudocounts")
            _check_test_method(test_method)
            
            minlen = _checkint(minlen,"minlen")
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
            maxbetween = _checkint(maxbetween,"maxbetween")
            
            _check_test_method(test_method)
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
            
            if save == None:
                
                return test2(expdata, background, TTAAframe, length, pvalue_cutoffbg = pvalue_cutoffbg, 
                        pvalue_cutoffTTAA = pvalue_cutoffTTAA,  mininser = min_hops, minlen = minlen,
                        extend = extend, maxbetween = maxbetween,  lam_win_size = lam_win_size,  
                        pseudocounts = pseudocounts, test_method = test_method, record = record )
            else:
                
                data = test2(expdata, background, TTAAframe, length, pvalue_cutoffbg = pvalue_cutoffbg, 
                        pvalue_cutoffTTAA = pvalue_cutoffTTAA,  mininser = min_hops, minlen = minlen,
                        extend = extend, maxbetween = maxbetween,  lam_win_size = lam_win_size,  
                        pseudocounts = pseudocounts, test_method = test_method, record = record )
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        elif method == "Blockify":
            
            print("For the Blockify method with background, [expdata, background, pvalue_cutoff, pseudocounts, test_method,  record] would be utilized.")
                
            if type(record) != bool:
                raise ValueError('Please enter a True/ False for record')
                
            _check_test_method(test_method)
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
            _checkint(pseudocounts,"pseudocounts")
            
            if save == None:
            
                return Blockify(expdata, background, length, pvalue_cutoff = pvalue_cutoff, pseudocounts = pseudocounts,
                            test_method = test_method , record = record)
            else:
                
                data = Blockify(expdata, background, length, pvalue_cutoff = pvalue_cutoff, pseudocounts = pseudocounts,
                            test_method = test_method , record = record)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        if method == "MACS2_old":
            
            print("For the MACS2 method with background, [expdata, background, reference, pvalue, lam_win_size, window_size, step_size,pseudocounts,  record] would be utilized.")
            
            if reference == "hg38":
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_hg38_ccf_new.bed",delimiter="\t",header=None)
            elif reference == "mm10":
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_mm10_ccf_new.bed",delimiter="\t",header=None)
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
            
            window_size = _checkint(window_size,"window_size")
            lam_win_size = _checkint(lam_win_size,"lam_win_size")
            step_size = _checkint(step_size,"step_size")
            
            if save == None:
            
                return callpeaksMACS2(expdata, background, TTAAframe, length, window_size = window_size, 
                                      lam_win_size=lam_win_size, step_size = step_size,
                                  pseudocounts = pseudocounts ,pvalue_cutoff = pvalue_cutoff, record = record)
            else:
                
                data = callpeaksMACS2(expdata, background, TTAAframe, length, window_size = window_size, 
                                      lam_win_size=lam_win_size, step_size = step_size,
                                  pseudocounts = pseudocounts ,pvalue_cutoff = pvalue_cutoff, record = record)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        else:
            
            raise ValueError("Not valid Method.")
            
            
    if background == None:
            
        
        if method == "MACS2":
            
            print("For the MACS2 method without background, [expdata, reference, pvalue_cutoff, lam_win_size, window_size, step_size, extend, pseudocounts, test_method, min_hops, record] would be utilized.")
            
            if reference == "hg38":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_hg38_ccf_new.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "mm10":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_mm10_ccf_new.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "yeast":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/yeast/yeast_S288C_dSir4_Background.ccf",delimiter="\t",header=None)
                length = 0
                
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
            
            window_size = _checkint(window_size,"window_size")
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size,"lam_win_size")
            extend = _checkint(extend,"extend")
            step_size = _checkint(step_size,"step_size")
            _checkint(pseudocounts,"pseudocounts")

            _check_test_method(test_method)
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
        
            if save == None:
                
                return callpeaksMACS2_bfnew2(expdata, TTAAframe, length, extend = extend, 
                      pvalue_cutoff =  pvalue_cutoff, window_size = window_size, 
                      lam_win_size = lam_win_size,  step_size = step_size, pseudocounts = pseudocounts,
                      test_method= test_method, min_hops = min_hops, record = record)
            else:
                
                data = callpeaksMACS2_bfnew2(expdata, TTAAframe, length, extend = extend, 
                      pvalue_cutoff =  pvalue_cutoff, window_size = window_size, 
                      lam_win_size = lam_win_size,  step_size = step_size, pseudocounts = pseudocounts,
                      test_method= test_method, min_hops = min_hops, record = record)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        elif method == "test":
            
            print("For the test method without background, [expdata, reference, pvalue_cutoff, lam_win_size, pseudocounts, minlen, extend, maxbetween, test_method, min_hops, record] would be utilized.")
            
            if reference == "hg38":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_hg38_ccf_new.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "mm10":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_mm10_ccf_new.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "yeast":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/yeast/yeast_S288C_dSir4_Background.ccf",delimiter="\t",header=None)
                length = 0
                
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
                
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size,"lam_win_size")
            extend = _checkint(extend,"extend")
            _checkint(pseudocounts,"pseudocounts")
            _check_test_method(test_method)
            
            minlen = _checkint(minlen,"minlen")
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
            maxbetween = _checkint(maxbetween,"maxbetween")
            
            _check_test_method(test_method)
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
            
        
            if save == None:
                
                return test_bf2(expdata, TTAAframe, length, pvalue_cutoff = pvalue_cutoff, mininser = min_hops, minlen = minlen,
                        extend = extend, maxbetween = maxbetween,  lam_win_size = lam_win_size,  
                        pseudocounts = pseudocounts, test_method = test_method, record = record )
            else:
                
                data = test_bf2(expdata, TTAAframe, length, pvalue_cutoff = pvalue_cutoff, mininser = min_hops, minlen = minlen,
                        extend = extend, maxbetween = maxbetween,  lam_win_size = lam_win_size,  
                        pseudocounts = pseudocounts, test_method = test_method, record = record )
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        elif method == "Blockify":
            
            print("For the Blockify method with background, [expdata, reference, pvalue_cutoff, pseudocounts, test_method,  record] would be utilized.")
            
            if reference == "hg38":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_hg38_ccf_new.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "mm10":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/scCCpackage_1/callPeaks/TTAA_mm10_ccf_new.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "yeast":
                
                TTAAframe = pd.read_csv("/scratch/rmlab/1/juanru/yeast/yeast_S288C_dSir4_Background.ccf",delimiter="\t",header=None)
                length = 0
                
            else:
                raise ValueError("Not valid reference.")
                
            _checkint(pseudocounts,"pseudocounts")

            if type(record) != bool:
                raise ValueError('Please enter a True/ False for record')
                
            _check_test_method(test_method)
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
            
            if save == None:
                
                return  Blockify(expdata, TTAAframe, length, pvalue_cutoff = pvalue_cutoff, pseudocounts = pseudocounts,
                            test_method = test_method , record = record)
            else:
                
                data =  Blockify(expdata, TTAAframe, length, pvalue_cutoff = pvalue_cutoff, pseudocounts = pseudocounts,
                            test_method = test_method , record = record)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        
        else:
            
            raise ValueError("Not valid Method.")

    else :
        
        raise ValueError("Not a valid background.")
  
            

def find_annotation_original(peakstartpoint, peakendpoint, startchrom, endchrom, refdata, refdataCUR):
      
    startFlag = 0
    endFlag = 0
    
    startreFlag = 0
    
    upstream = []
    middle = []
    downstream = []
    
    minnumber = 1000000000000000
    
    for i in range(len(refdata[:,0])):
        
        if refdata[i,1] > startreFlag and refdata[i,1] < peakstartpoint:
            startreFlag = refdata[i,1]
            upstream = [i] 
            
        elif refdata[i,1] == startreFlag:
            upstream.append(i)
        
            
        if (refdata[i,0] < peakstartpoint and refdata[i,1] > peakstartpoint) or (refdata[i,0] < peakendpoint and refdata[i,1] > peakendpoint):
            middle.append(i)
            minnumber = min(minnumber, i)
            
        if refdata[i,0] > peakendpoint:
            if endFlag == 0:
                minnumber = min(minnumber, i)
                downstream.append(i)
                endFlag = 1
            else:
                if refdata[i,0] == refdata[i-1,0]:
                    downstream.append(i)
                else:
                    break

    if upstream == []:
        upstream = None
    else:
        minnumber = min(minnumber, min(upstream))
        upstream = refdataCUR[upstream]
        
    if middle == []:
        middle = None
    else:
        middle = refdataCUR[middle]
    
    if downstream == []:
        downstream = None
    else:
        downstream = refdataCUR[downstream]
        
        
    return upstream, middle, downstream, refdata[max(minnumber-15,1):,:], refdataCUR[max(minnumber-15,1):,:]


@jit(nopython=True)
def find_annotation_numba(peakstartpoint, peakendpoint, startchrom, endchrom, refdata, refdataCUR):
    
    startFlag = 0
    endFlag = 0
    
    startreFlag = 0
    
    upstreamFlag = 0
    middleFlag = 0
    downstreamFlag = 0
    
    minnumber = None
    minnumber = 1000000000000000
    
    for i in range(len(refdata[:,0])):
        
        if refdata[i,1] > startreFlag and refdata[i,1] < peakstartpoint:
            startreFlag = refdata[i,1]
            upstream = [i] 
            
        elif refdata[i,1] == startreFlag:
            upstream.append(i)
        
        #if  i>0 and refdata[i,1] == refdata[i-1,1] and startFlag == 0:
        #    startreFlag = i-1
        #elif refdata[i,1] > peakstartpoint and startFlag == 0:
        #    upstreamFlag = 1
        #    upstream = list(range(startreFlag,max(i,1)))
        #    minnumber = min(minnumber, min(upstream))
        #    startFlag = 1
        #elif startFlag ==0:
        #    startreFlag = i
            
        if (refdata[i,0] < peakstartpoint and refdata[i,1] > peakstartpoint) or (refdata[i,0] < peakendpoint and refdata[i,1] > peakendpoint):
            if middleFlag == 0:
                middle = [i]
                middleFlag = 1
            else:
                middle.append(i)
            
            minnumber = min(minnumber, i)
            
        if refdata[i,0] > peakendpoint:
            if endFlag == 0:
                minnumber = min(minnumber, i)
                downstreamFlag = 1
                downstream = [i]
                endFlag = 1
            else:
                if refdata[i,0] == refdata[i-1,0]:
                   
                    downstream.append(i)
                else:
                    break
                    
        
    if upstreamFlag == 0:
        upstream = None
    else:
        upstream = refdataCUR[np.array(upstream)]
        minnumber = min(minnumber, min(upstream))
        
    if middleFlag == 0:
        middle = None
    else:
        middle = refdataCUR[np.array(middle)]
        
    if downstreamFlag == 0:
        downstream = None
    else:
        downstream = refdataCUR[np.array(downstream)]

    return upstream, middle, downstream, refdata[max(minnumber-1,1):,:], refdataCUR[max(minnumber-1,1):,:]


def _myFuncsorting(e):
    try:
        return int(e.split('_')[0][3:])
    except :
        return int(ord(e.split('_')[0][3:]))

# pair function 
def _peakToCell_function(insertionschr, peakschr, chromo, cellpeaks, peak_name_dict, barcodes_dict):
    peak_id = 0
    
    for insertion in range(len(insertionschr)):
        
        if (insertionschr.iat[insertion,1] >= peakschr.iat[peak_id,1]-3) and insertionschr.iat[insertion,2] <= (peakschr.iat[peak_id,2]+3):
            cellpeaks[peak_name_dict[chromo+"_"+str(peakschr.iat[peak_id,1])+"_"+str(peakschr.iat[peak_id,2])],barcodes_dict[insertionschr.iat[insertion,5]]]+= 1
        elif peak_id +1 == len(peakschr):
            break
        elif insertionschr.iat[insertion,1] >= peakschr.iat[peak_id+1,1]-3 and insertionschr.iat[insertion,2] <= peakschr.iat[peak_id+1,2]+3:
            cellpeaks[peak_name_dict[chromo+"_"+str(peakschr.iat[peak_id+1,1])+"_"+str(peakschr.iat[peak_id+1,2])],barcodes_dict[insertionschr.iat[insertion,5]]]+= 1
            peak_id = peak_id +1

    return cellpeaks

 
    
def makeAnndata(
    insertions: pd.DataFrame, 
    peaks: pd.DataFrame, 
    barcodes: pd.DataFrame
    ) -> AnnData:
    
    try:
        barcodes = list(barcodes[0])
    except:
        barcodes = list(barcodes["index"])
    barcodes_dict = {}
    for i in range(len(barcodes)):
        barcodes_dict[barcodes[i]] = i

    # peaks to order
    
    try:
        peak_name = peaks[["Chr","Start","End"]].T.astype(str).agg('_'.join)
    except ValueError:
        peak_name = peaks[["0,1,2"]].T.astype(str).agg('_'.join)
        

    peak_name_unique = list()
    for number in peak_name:
        if number not in peak_name_unique:
            peak_name_unique.append(number)
    peak_name_unique.sort(key=_myFuncsorting)

 
    peak_name_dict = {}
    for i in range(len(peak_name_unique)):
        peak_name_dict[peak_name_unique[i]] = i
        

  
    # create an empty matrix to store peaks * cell
    cellpeaks = lil_matrix((len(peak_name_unique),len(barcodes)), dtype=np.float32)
    
    
    #pairing
    try:
        chrolist =  list(peaks["Chr"].unique())
    except ValueError:
        chrolist =  list(peaks[0].unique())
        
    for chromosome in tqdm.tqdm(chrolist):

        try:
            insertionschr = insertions[insertions["Chr"] == chromosome]
        except :
            insertionschr = insertions[insertions[0] == chromosome]
            
        try:
            peakschr = peaks[peaks["Chr"] == chromosome]
        except :
            peakschr = peaks[peaks[0] == chromosome]

        cellpeaks = _peakToCell_function(insertionschr, peakschr, chromosome, cellpeaks, peak_name_dict, barcodes_dict)
        
    peaks.index = peak_name
    adata = ad.AnnData(csr_matrix(cellpeaks).T,obs= pd.DataFrame(index=barcodes),var = peaks)
    
    return adata
        
        
    
def combine_annotation(peak_data, peak_annotation):
    
    try:
        peak =  peak_data[["Chr","Start","End"]]
    except:
        peak =  peak_data[[0,1,2]]
        
    try:
        annotation =  peak_annotation[["Chr","Start","End"]]
    except:
        annotation =  peak_annotation[[0,1,2]]
        
    if ((peak == annotation).all()).all():
        return pd.concat([peak_data, peak_annotation.iloc[:,3:]],axis = 1)
    else:
        print("The peaks for peak data and anotation data are not the same")
        
def annotation(peaks_frame = None, peaks_path = None, reference = "hg38", method = "homer", save_peak = None, save_annotation = None, bedtools_path = None):
    
    if method == "homer":
        
        import os
        print("In homer method, we would use homer under your default path.")
        
        if  peaks_path != None:
            print("We would use peaks in the peaks_path file and temporary anotation file would be saved to save_annotation.")
            
            save_peakinitial = save_peak 
            save_annotationinitial = save_annotation
            if save_peak ==  None:
                import random
                save_peak = "temp_peaks_"+str(random.randint(1,1000))+".gos"
                
            pd.read_csv(peaks_path,sep = "\t", header = None)[[0,1,2]].to_csv(save_peak, sep = "\t",
                                                                  header = None, index = None)
            
            if save_annotation ==  None:
                import random
                save_annotation = "save_annotation_"+str(random.randint(1,1000))+".gos"
                
            cmd = "annotatePeaks.pl " + save_peak + " " + reference + " > " + save_annotation
            
        elif type(peaks_frame) == pd.DataFrame :
            print("Temporary peak would be saved to save_peak and temporary anotation file would be saved to save_annotation.")
            
            if save_peak ==  None:
                import random
                save_peak = "temp_peaks_"+str(random.randint(1,1000))+".gos"
                
                
            peaks_frame[[0,1,2]].to_csv(save_peak, sep = "\t", index = None, header = None)
            
            if save_annotation ==  None:
                import random
                save_annotation = "save_annotation_"+str(random.randint(1,1000))+".gos"

            cmd = "annotatePeaks.pl " + save_peak + " " + reference + " > " + save_annotation
            
        else:
             print("Please input a valid peak.")
            
        os.system(cmd)
        annotated_peaks = pd.read_table(save_annotation, index_col = 0).sort_values(["Chr","Start"])
        
        if save_annotationinitial ==  None:
            os.remove(save_annotation)
            
        if save_peakinitial == None:
            os.remove(save_peak)
            
        return annotated_peaks[["Chr","Start","End","Nearest Refseq","Gene Name"]]
    
    elif method == "bedtools":
        
        import pybedtools
        print("In the bedtools method, we would use bedtools in the default path. Set bedtools path by 'bedtools_path' if needed.")
        
        if bedtools_path != None:
            pybedtools.helpers.set_bedtools_path(path=bedtools_path)
            
            
        if  peaks_path != None:
            peaks_bed = pybedtools.BedTool(peaks_path)
            
            
        elif type(peaks_frame) == pd.DataFrame :
            
            save_peakinitial = save_peak
            if save_peak ==  None:
                
                import random
                save_peak = "temp_peaks_"+str(random.randint(1,1000))+".bed"
                
            peaks_frame.to_csv(save_peak, sep = "\t", index = None, header = None)
            peaks_bed = pybedtools.BedTool(save_peak)
            
        
        else :
            print("Please input a valid peak.")
            
        if reference == "hg38":
            refGene_filename = pybedtools.BedTool("/scratch/ref/rmlab/calling_card_ref/human/refGene.hg38.Sorted.bed")
        elif reference == "mm10":
            refGene_filename = pybedtools.BedTool("/scratch/ref/rmlab/calling_card_ref/mouse/refGene.mm10.Sorted.bed")
            
        temp_annotated_peaks = peaks_bed.closest(refGene_filename,D="ref",t="first",k=2)
        
        temp_annotated_peaks = pd.read_table(temp_annotated_peaks.fn, header = None).iloc[: ,[0,1,2,-4,-3 ]]
        temp_annotated_peaks = temp_annotated_peaks
        temp_annotated_peaks.columns = ["Chr","Start","End","Nearest Refseq","Gene Name"]
        temp_annotated_peaks1 = temp_annotated_peaks.iloc[::2].reset_index()
        temp_annotated_peaks1 = temp_annotated_peaks1[["Chr","Start","End","Nearest Refseq",
                                                       "Gene Name"]].rename(columns={"Nearest Refseq": "Nearest Refseq1", 
                                                                                     "Gene Name": "Gene Name1"})
        temp_annotated_peaks2 = temp_annotated_peaks.iloc[1::2].reset_index()
        temp_annotated_peaks2 = temp_annotated_peaks2[["Nearest Refseq",
                                                       "Gene Name"]].rename(columns={"Nearest Refseq": "Nearest Refseq2", 
                                                                                     "Gene Name": "Gene Name2"})

        finalresult = pd.concat([temp_annotated_peaks1, temp_annotated_peaks2],axis = 1)
        
        if save_annotation != None:
            finalresult.to_csv(save_annotation,index = None, sep = "\t")
            
        if save_peakinitial ==  None:
            import os
            os.remove(save_peak)
                
                
        return pd.concat([temp_annotated_peaks1, temp_annotated_peaks2],axis = 1)

_Method_rank_cc_groups = Optional[Literal['binomtest', 'binomtest2','fisher_exact', 'chisquare']]
    
def calculatePvalue(number1,number2,total1,total2,method = "binomtest"):
    
    if method == "binomtest":
        
        from scipy.stats import binomtest
        
        return binomtest(int(number1), n=int(number1+number2), p=float(total1/(total1+total2))).pvalue

    if method == "binomtest2":
        
        from scipy.stats import binomtest
        
        return max(binomtest(int(number1), n=total1, p=number2/total2).pvalue, 
                   binomtest(int(number2), n=total2, p=number1/total1).pvalue)
    
    elif method == "fisher_exact":
        
        table = np.array([[number1, number2], [total1- number1, total2 - number2]])
        from scipy.stats import fisher_exact
        _,p = fisher_exact(table, alternative='two-sided')
        
        return p
    
    elif method == "chisquare":
        
        from scipy.stats import chisquare
        ratio = (number1+number2) /(total1+total2)
        
        return chisquare([number1, number2], f_exp=[ratio*total1, ratio*total2]).pvalue
    
    else:
        
        raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact/chisquare.")
        
        

def diff2group(
    adata_ccf: AnnData,
    name1: str,
    name2: str, 
    peakname: Optional[str] = None ,
    test_method: _Method_rank_cc_groups = "binomtest"
) -> Union[list[float], float]:
    
    if peakname != None:

        cluster1 = adata_ccf[(adata_ccf.obs[["cluster"]] == name1)["cluster"], 
                             adata_ccf.var.index.get_loc(peakname)].X
        cluster2 = adata_ccf[(adata_ccf.obs[["cluster"]] == name2)["cluster"], 
                             adata_ccf.var.index.get_loc(peakname)].X

        total1 = cluster1.shape[0]
        total2 = cluster2.shape[0]

        if test_method == "binomtest2" or test_method == "fisher_exact":
            number1 = cluster1.nnz
            number2 = cluster2.nnz
        elif test_method == "binomtest" or test_method == "chisquare":
            number1 = cluster1.sum()
            number2 = cluster2.sum()
        else:
            raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact/chisquare.")

        return calculatePvalue(number1,number2,total1,total2,method = test_method)
        
    else:
        
        print("No peak name is provided, the pvalue for all the peaks would be returned.")
        pvaluelist = []
        
        for peak in list(adata_ccf.var.index):

            
            cluster1 = adata_ccf[(adata_ccf.obs[["cluster"]] == name1)["cluster"], 
                             adata_ccf.var.index.get_loc(peak)].X
            cluster2 = adata_ccf[(adata_ccf.obs[["cluster"]] == name2)["cluster"], 
                                 adata_ccf.var.index.get_loc(peak)].X

            total1 = cluster1.shape[0]
            total2 = cluster2.shape[0]

            if test_method == "binomtest2" or test_method == "fisher_exact":
                number1 = cluster1.nnz
                number2 = cluster2.nnz
            elif test_method == "binomtest" or test_method == "chisquare":
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact/chisquare.")
            
            pvaluelist.append(calculatePvalue(number1,number2,total1,total2,method = test_method))
            
        return pvaluelist
            



def rank_cc_groups(
    adata_ccf: AnnData, 
    groupby: str,
    use_raw: bool = True,
    groups: Union[Literal['all'], Iterable[str]] = 'all',
    reference: str = None,
    n_cc: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    method: _Method_rank_cc_groups = None
) -> Optional[AnnData]:
       
      
    avail_method = ['binomtest', 'binomtest2','fisher_exact', 'chisquare']
    if method == None:
        method = 'binomtest'
    elif method not in avail_method:
        raise ValueError(f'Correction method must be one of {avail_method}.')
        
    possible_group = list(adata_ccf.obs[groupby].unique())

    if reference == None:
        reference = "rest"
    elif reference not in possible_group:
        raise ValueError(f'Invalid reference, should be all or one of {possible_group}.')
        
    if groups == 'all':
        group_list = possible_group
    elif type(groups) == str: 
        group_list = groups
    elif type(groups) == list:
        group_list = groups
    else:
        raise ValueError("Invalid groups.")
        
    if key_added == None:
        key_added = 'rank_cc_groups'
    elif type(key_added) != str:
        raise ValueError("key_added should be str.")
        
    if type(use_raw) != bool:
        print( "use_row should be bool.")
    
    adata_ccf = adata_ccf.copy() if copy else adata_ccf
    
    adata_ccf.uns[key_added] = {}
    adata_ccf.uns[key_added]['params'] = dict(
    groupby=groupby,
    reference=reference,
    method=method,
    use_raw=use_raw)
    
    peak_list  = list(adata_ccf.var.index)

    if n_cc == None:
        n_cc = len(peak_list)
    elif type(n_cc) != int or n_cc < 1 or n_cc > len(peak_list):
        raise ValueError("n_cc should be a int larger than 0 and smaller than the total number of peaks ")


    finalresult_name = np.empty([n_cc, len(group_list)], dtype='<U100')
    finalresult_pvalue = np.empty([n_cc, len(group_list)], dtype=float)

    i = 0
    
    for cluster in group_list:
        
        if reference == "rest":
            clusterdata = adata_ccf[(adata_ccf.obs[["cluster"]] == cluster)["cluster"]]
            clusterdatarest = adata_ccf[(adata_ccf.obs[["cluster"]] != cluster)["cluster"]]
        else:
            clusterdata = adata_ccf[(adata_ccf.obs[["cluster"]] == cluster)["cluster"]]
            clusterdatarest = adata_ccf[(adata_ccf.obs[["cluster"]] == reference)["cluster"]]

            
        
        pvaluelist = []
        
        for peak in peak_list:
            
            cluster1 = clusterdata[:,adata_ccf.var.index.get_loc(peak)].X
            total1 = cluster1.shape[0]

            cluster2 = clusterdatarest[:,adata_ccf.var.index.get_loc(peak)].X
            total2 = cluster2.shape[0]

            if method == "binomtest2" or method == "fisher_exact":
                number1 = cluster1.nnz
                number2 = cluster2.nnz
            elif method == "binomtest" or method == "chisquare":
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact/chisquare.")

            
            
  
            pvaluelist.append(calculatePvalue(number1,number2,total1,total2,method =  method))
            
        pvaluelistnp = np.array(pvaluelist)
        pvaluelistarg = pvaluelistnp.argsort()
        
        
        finalresult_name[:,i] = np.array(peak_list)[pvaluelistarg][:n_cc]
        finalresult_pvalue[:,i] = pvaluelistnp[pvaluelistarg][:n_cc]

        i += 1
     

    temppvalue = np.array([group_list, ['float']*len(group_list)]).transpose()
    tempname = np.array([group_list, ['<U100']*len(group_list)]).transpose()

  
    adata_ccf.uns[key_added]['names'] = np.rec.array(list(map(tuple, finalresult_name)), dtype=list(map(tuple, tempname))) 
    adata_ccf.uns[key_added]['pvalues'] = np.rec.array(list(map(tuple, finalresult_pvalue)), dtype=list(map(tuple, temppvalue))) 
    
    
    
    return adata_ccf if copy else None


def plot_rank_cc_groups(
    adata_ccf: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_cc: int = 10,
    cc_symbols: Optional[str] = None,
    key: Optional[str] = 'rank_cc_groups',
    fontsize: int = 8,
    ncols: int = 4,
    sharey: bool = True,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    ax: Optional[Axes] = None,
    **kwds,
):
    if 'n_panels_per_row' in kwds:
        n_panels_per_row = kwds['n_panels_per_row']
    else:
        n_panels_per_row = ncols
    if n_cc < 1:
        raise NotImplementedError(
            "Specifying a negative number for n_cc has not been implemented for "
            f"this plot. Received n_cc={n_cc}."
        )

    reference = str(adata_ccf.uns[key]['params']['reference'])
    group_names = adata_ccf.uns[key]['names'].dtype.names if groups is None else groups
    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)

    from matplotlib import gridspec

    fig = pl.figure(
        figsize=(
            n_panels_x * rcParams['figure.figsize'][0],
            n_panels_y * rcParams['figure.figsize'][1],
        )
    )
    gs = gridspec.GridSpec(nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3)

    ax0 = None
    ymin = np.Inf
    ymax = -np.Inf
    for count, group_name in enumerate(group_names):
        gene_names = adata_ccf.uns[key]['names'][group_name][:n_cc]
        pvalues = adata_ccf.uns[key]['pvalues'][group_name][:n_cc]

        # Setting up axis, calculating y bounds
        if sharey:
            ymin = min(ymin, np.min(pvalues))
            ymax = max(ymax, np.max(pvalues))

            if ax0 is None:
                ax = fig.add_subplot(gs[count])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[count], sharey=ax0)
        else:
            ymin = np.min(pvalues)
            ymax = np.max(pvalues)
            ymax += np.min(0.3 * (ymax - ymin),1)

            ax = fig.add_subplot(gs[count])
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(-0.9, n_cc - 0.1)

        # Mapping to cc_symbols
        if cc_symbols is not None:
            if adata_ccf.raw is not None and adata_ccf.uns[key]['params']['use_raw']:
                gene_names = adata_ccf.raw.var[cc_symbols][gene_names]
            else:
                gene_names = adata_ccf.var[cc_symbols][gene_names]

        # Making labels
        for ig, gene_name in enumerate(gene_names):
            ax.text(
                ig,
                pvalues[ig],
                gene_name,
                rotation='vertical',
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=fontsize,
            )

        ax.set_title('{} vs. {}'.format(group_name, reference))
        if count >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel('ranking')

        # print the 'score' label only on the first panel per row.
        if count % n_panels_x == 0:
            ax.set_ylabel('pvalue')

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)

    writekey = f"rank_cc_groups_{adata_ccf.uns[key]['params']['groupby']}"
    savefig_or_show(writekey, show=show, save=save)
    
    
def savefig_or_show(
    writekey: str,
    show: Optional[bool] = None,
    dpi: Optional[int] = None,
    ext: str = None,
    save: Union[bool, str, None] = None,
):
    if isinstance(save, str):
        # check whether `save` contains a figure extension
        if ext is None:
            for try_ext in ['.svg', '.pdf', '.png']:
                if save.endswith(try_ext):
                    ext = try_ext[1:]
                    save = save.replace(try_ext, '')
                    break
        # append it
        writekey += save
        save = True
    save = False if save is None else save
    show = False if show is None else show
    if save:
        savefig(writekey, dpi=dpi, ext=ext)
    if show:
        pl.show()
    if save:
        pl.close()  # clear figure


        
