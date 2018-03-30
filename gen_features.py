#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 01 11:03:40 2013

@author: sv507
"""
from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import collections



def save_sparse(filename,matrix):
    np.savez(filename,data=matrix.data,indices=matrix.indices,indptr=matrix.indptr)

def csr_write_libsvm(filename,X_csr, y, nfeatures):
    """ write to libsvm format
    the CSR matrix is assumed to be 'regular' nrows x nfeatures
    ( so eg a feature with zero value should also be included)
    """
    #assert X_csr.shape[0]==y.shape[0]
    #assert X_csr.shape[0]*nfeatures==X_csr.indices.shape[0]
    with open(filename,'wb') as fp:
        for d in zip(y,X_csr.indices.reshape(-1,nfeatures),
                     X_csr.data.reshape(-1,nfeatures)):
            print >>fp,'{}'.format(d[0]),
            for d1 in zip(d[1]+1,d[2]) :
                if d1[1]: print >>fp,'{:.0f}:{:.0f}'.format(*d1),
            print >>fp
        # add 1 to d[1] because libsvm requires 1 based index

def csr_subrows(X_csr,indices, nfeatures):
    """ ? not sure if necessary supposed to be for removing columns
        the CSR matrix is assumed to be 'regular' nrows x nfeatures
    ( so eg a feature with zero value should also be included)
    """

    return csr_matrix((
        X_csr.data.reshape(-1,nfeatures)[indices,:].ravel(),
        X_csr.indices.reshape(-1,nfeatures)[indices,:].ravel(),
        nfeatures*np.arange(len(indices)+1)))
            

def make_factor_table(dataframe,fac_name,CTR):
    gp=dataframe.groupby(fac_name)
    if 'instances' in dataframe.columns:
        fac_df = gp[['instances','clicks']].sum()
        fac_df.rename(columns={'instances':'count'},inplace=True)
    else:        
        fac_df=gp['clicks'].aggregate(['mean','count'])
    
    fac_df['stderr']=np.sqrt(CTR*(1-CTR)/fac_df['count'])
    fac_df['nstderr']=abs(fac_df['mean']-CTR)/fac_df['stderr']
    fac_df['flstderr']=(fac_df['count']*CTR>10)*fac_df['nstderr']
    fac_df.sort('count',ascending=False,inplace=True)
    return fac_df

def calc_CTR(group):
        d = group['clicks']
        w = group['count']
        # return two values count and CTR 
        return pd.Series({'count':w.sum(),'mean':(d * w).sum() / w.sum()})    
    
    fac_df=gp.apply(calc_CTR)
    CTR=np.inner(fac_df['mean'],fac_df['count'])/fac_df['count'].sum()
    fac_df['stderr']=np.sqrt(CTR*(1-CTR)/fac_df['count'])
    fac_df['nstderr']=abs(fac_df['mean']-CTR)/fac_df['stderr']
    fac_df['flstderr']=(fac_df['count']*CTR>10)*fac_df['nstderr']
# te=pd.DataFrame({'a':[0,1,0],'clicks':[0,0,1],'b':[1,2,3]})
    # wts=np.array([1,3,5]).reshape(3,)
# d1,c1=make_factor_table_weighted(te,'a',wts)  

  
def make_factor_table_weighted(dataframe,fac_name,counts_name, events_name):
    # http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns   
    weights_df=dataframe[fac_name] # need to deal with list of names
    weights_df['counts']=weights
    weights_df['events']=dataframe['clicks']
    # hope concat doesn't create new [deep copy] dataframe
    
    fac_df=weights_df.groupby(fac_name)['count'].sum()
    fac_df.sort('counts',ascending=False,inplace=True)
    return fac_df,CTR
    

def make_factor_table_weighted_test():
	te = pd.DataFrame({'a':[0,1,0],
				  'clicks':[0,0,1],
				       'b':[1,2,3]})
    wts=np.array([1,3,5]).reshape(3,)
	d1, c1 = make_factor_table_weighted(te, 'a', wts)
	
# take id metrics (eg clicks) etc
# so sum all whichever variable we are 'counting'

class SparseCat:
    def __init__(self,factors,non_factors ):
        # a list of named features ("site",...,'site*ad')
        # make copy
        self.factors=tuple(factors)
        self.non_factors=tuple(non_factors)
    
    #def get_params():
    #def set_params(**params):
        
    def fit(self,X,y):
        # take pandas array and convert to csr representation
        self.factors_table_=dict()        
        
        for f in self.factors:
            fs=f.split('*') # either 'ad' or 'ad*site*device' etc
            # keep ordering or not? no checking of duplicates etc
            # why use y at all? X has to contain clicks anyway
            self.factors_table_[f]=make_factor_table(X,fs,self.click_rate_)
    
    def fit_weighted(self,X,weights):
        # take pandas array and convert to csr representation
        self.factors_table_=dict()        
        
        for f in self.factors:
            fs=f.split('*') # either 'ad' or 'ad*site*device' etc
            # keep ordering or not? no checking of duplicates etc
            self.factors_table_[f],self.click_rate_=make_factor_table_weighted(X,fs,weights)

    def set_params(self,count_cutoff):
        self.count_cutoff=count_cutoff
                  
    def transform(self,X):
        ndata=X.shape[0] # original data length
        # idea is that we set "irrel" to zero and all other
        self.mappings_=dict()
        
        for f in self.factors:
            df=self.factors_table_[f]
            sig_levels=df[df['count']>self.count_cutoff].index
            # we add 1 to tell whether item exists or not in mappings
            # identify feature levels with flnst>2
            # idea is either we find in dict and set corresponding point to 1,
            # or (initialised) data is left at zero
            self.mappings_[f]=collections.defaultdict(int,zip(sig_levels,range(1,len(sig_levels)+1)))
        
        #identify NA !!!
        self.len_factors= [ len(self.mappings_[factor]) for factor in self.factors]

        nfeatures=len(self.factors)+len(self.non_factors) #
        
        ncols=sum(self.len_factors)+len(self.non_factors) # # dummy variables + other
        
        
        start_non_factor=sum(self.len_factors)
        # should handle lim cases of zero factor/ non factor
        start_features=np.concatenate(([0],
                                       np.cumsum(self.len_factors[:-1]),
                                np.arange(start_non_factor,start_non_factor+len(self.non_factors))))
        
        indices=np.zeros((ndata,nfeatures),dtype=np.int)
        vals=np.ones((ndata,nfeatures))
        for ifactor,factor in enumerate(self.factors):
            factor_split=factor.split('*')
            # mult or single !!!
            if len(factor_split)==1:
                index=X[factor].map(self.mappings_[factor])
            else:
                index=X[factor_split].apply(lambda x: self.mappings_[factor][x],axis=1)
                

            # index starts at 1 to identify missing fields as zero
            index_zeros=np.where(index==0)[0]
            indices[:,ifactor]= index - 1 + start_features[ifactor]
            # now fix the few zero entries
            indices[index_zeros]=start_features[ifactor]
            vals[index_zeros,ifactor]=0

        for i,non_factor in enumerate(self.non_factors):
            indices[:,len(self.factors)+i]=start_non_factor+i
            vals[:,len(self.factors)+i]=X[non_factor]
        
        X_sparse=csr_matrix((vals.ravel(),indices.ravel(),nfeatures*np.arange(ndata+1)))
        X_sparse.has_sorted_indices = True        
        return X_sparse


#if __name__ == "__main__":
#    
#    
#    
#    sc_new=SparseCat(factors,non_factors)
#    sc_new.fit(small,small['clicks'])
#    sc_new.set_params(count_cutoff=10)
#    z_new=sc_new.transform(small)
