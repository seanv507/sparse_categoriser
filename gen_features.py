#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 08:57:04 2018

@author: sviolante
"""

import pandas as pd
import numpy as np
import scipy.sparse

def make_cv_df(cv_list):
    df=pd.DataFrame.from_items([('C:{0}'.format(c),clf.grid_scores_[i][2]) for i,c in enumerate(clf.param_grid[0]['C'])])


def csr_write_libsvm(filename,X_csr, y,nfeatures):
    with open('filename','wb') as fp:
        for d in zip(y,X_csr.indices.reshape(-1,nfeatures),X_csr.data.reshape(-1,nfeatures)):
            print >>fp,'{}'.format(d[0]),
            for d1 in zip(d[1],d[2]) :
                if d1[1]: print >>fp,'{:.0f}:{:.0f}'.format(*d1),
            print >>fp


def make_factor_table(dataframe, fac_name, hit_col, CTR):
    gp = dataframe.groupby(fac_name)
    fac_df=gp[hit_col].aggregate(['mean','count'])
    fac_df['stderr'] = np.sqrt(CTR * (1 - CTR) / fac_df['count'])
    fac_df['nstderr']=abs(fac_df['mean'] - CTR) / fac_df['stderr']
    fac_df['flstderr']=(fac_df['count'] * CTR > 10) * fac_df['nstderr']
    fac_df.sort_values('count', ascending=False, inplace=True)
    return fac_df


class SparseCategoriser:
    def __init__(self, factors, non_factors, hit_col ):
        # a list of named features ("site",...,'site*ad')
        # make copy
        self.factors = tuple(factors)
        self.non_factors = tuple(non_factors)
        self.hit_col = hit_col
        self.count_cutoff = 0

    #def get_params():
    #def set_params(**params):

    def fit(self, X, y):
        # take pandas array and convert to csr representation
        self.factors_table_ = dict()
        self.hit_rate_ = np.mean(y)
        for f in self.factors:
            fs = f.split('*') # either 'ad' or 'ad*site*device' etc
            # keep ordering or not? no checking of duplicates etc
            self.factors_table_[f] = make_factor_table(X, fs, self.hit_col, self.hit_rate_)
        return self

    def set_params(self, count_cutoff):
        self.count_cutoff = count_cutoff

    def transform(self, X):
        ndata = X.shape[0] # original data length
        # idea is that we set "irrel" to zero and all other
        self.mappings_ = dict()

        for f in self.factors:
            df = self.factors_table_[f]
            sig_levels = df[df['count'] > self.count_cutoff].index

            # identify feature levels with flnst>2
            # idea is either we find in dict and set corresponding point to 1,
            # or (initialised) data is left at zero
            self.mappings_[f]=dict(zip(sig_levels,range(len(sig_levels))))

        #identify NA !!!
        self.len_factors = list(map(len,self.mappings_.values()))
        nfeatures = len(self.factors) + len(self.non_factors) #

        ncols = sum(self.len_factors) + len(self.non_factors) # # dummy variables + other

        start_non_factor = sum(self.len_factors)
        # should handle lim cases of zero factor/ non factor
        start_features=np.concatenate(([0],
            np.cumsum(self.len_factors[:-1]),
            np.arange(start_non_factor,
            start_non_factor + len(self.non_factors))))
        # we initialise indices because we have zero values too (ie if we don't specify a feature it =0 )
        indices=np.tile(start_features,(ndata,1))
        vals=np.zeros((ndata,nfeatures))
        for ifactor, factor in enumerate(self.factors):
            # mult or single !!!
            for feat_val, ifeat_val in self.mappings_[factor].items():
                factor_split = factor.split('*')
                if len(factor_split) == 1:
                    lrows = X[factor] == feat_val
                else:
                    lrows = X.apply(
                        lambda x: reduce(lambda v1, v2: v1 & v2,
                           [(x[colname] == value)  for colname, value
                               in zip(factor_split,feat_val)]
                        ), axis=1)


                indices[lrows,ifactor] = ifeat_val + start_features[ifactor]
                vals[lrows,ifactor] = 1


        for i, non_factor in enumerate(self.non_factors):
            indices[:, len(self.factors) + i] = start_non_factor + i
            vals[:, len(self.factors) + i] = X[non_factor]

        X_sparse = scipy.sparse.csr_matrix((vals.ravel(),
                                            indices.ravel(),
                                            nfeatures*np.arange( ndata + 1)))
        return X_sparse


if __name__ == "__main__":



    #mad=pd.read_table('full.tsv',dtype={
    #'token':object,	'site':object,	'ad':object,	'country':object,	'rtb':bool,	'acct':object,
    #'campaign':object,	'banner':int,	'spotbuy':object,	'appid':object,	'device':object,	'token1':object,	'hits':int,	'conversions':int})
    # mad.fillna('NA',inplace=True)

    factors=['ad','campaign','acct','site','country','device']
    non_factors=['rtb','banner']


#    hit_rate=small['hits'].mean()
#
#    sc=SparseCategoriser(factors,non_factors)
#    sc.fit(small,small['hits'])
#    sc.set_params(count_cutoff=10)
#    z=sc.transform(small)
