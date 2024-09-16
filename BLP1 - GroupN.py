# =============================================================================
#
#    COMPUTATIONAL ASSIGNMENT
#
# We will be required to estimate a random coefficient logit model (i.e. BLP algorithm) for the app market.  
#
# =============================================================================
# =============================================================================
#    data :number \$
#         
#         demogr:   demographic variables with (T by ns*D) dimension
#         X1 :     product characteristics
#         X2 :     subset of X1 that gets random coefficients by interacting with 
#         the demographic and noise distributions 
#
#         X1_names, X2_names, D_names: names of variables from X1, X2
#                  and demographic variables respectively
#        
#         IV  :     Instruments with (TJ by K) dimension 
#         s_jt:    Observed market shares of brand j in market t ; (TJ by 1) dimension 
#         cdindex: Index of the last observation for a market (T by 1)
#         cdid:    Vector of market indexes (TJ by 1)
#         cdid_demogr: vector assigning market to each demographic observation
#         ns: number of simulated "indviduals" per market 
#         nmkt: number of markets
#         K1: number of product characteristics/elements in x1 
#         k2: number of elements in x2
#         TJ is number of observations (TJ = T*J if all products are observed in all markets)
#         v  :     Random draws given for the estimation with (T by ns*(k1+k2)) dimension
#         Random draws. For each market ns*K2 iid normal draws are provided.
#         They correspond to ns "individuals", where for each individual
#         there is a different draw for each column of x2. 
#
# =============================================================================


import pandas as pd
import numpy as np
import scipy.stats as stats


a = 0


# Import the dataset for the app market
df = pd.read_csv("dataset_computational_noise.csv")
df['constant'] = 1

#Drop if there are missing values
df = df.dropna(subset =['averagescore'], how = 'all')
df.reset_index(drop = True, inplace = True)

# Instrument for IV. We will use the number of apps in each category as an instrument since we do not have cost shifters (i.e. variables that shift costs but do not shift demand).  
df['group_n'] = df.groupby(['country','newgroup','year'])['constant'].transform('sum')

# Compute Market Shares: markets are defined by country/year
df['new_est_total'] = df.groupby(['country','year'])['new_est'].transform('sum')     # Sum of downloads in each market
df['s_jt'] = df.new_est/df.new_est_total                                             # Market shares of each product for every market

# Drop if Market Share < 0.1%
df = df[df.s_jt > 0.001]
df.reset_index(drop = True, inplace = True)

# Parameters for population and rescaling 
ns = 500                                                                        # number of ppl in each market
scale = 100                                                            # Used to rescale the data later
m = len(set(list(df.year))) *len(set(list(df.country)))                         # number of markets = years*countries
#v = pd.DataFrame(np.random.randn(m,ns*2))                                       # In our case we only use three variables so k2+1 = 2
                                                                                # v is a random draw that represents consumer heterogeneity
#Print v to compare different intial guess with saem reandom drawn (Isac)
#v.to_csv('v_rand1.csv', index=False, header=False)
v = pd.read_csv('v_rand1.csv', names= list(range(0,1000)))

# Demographics
demogr = pd.read_csv('demogr_apps.csv', names= list(range(1,1001)))             # Import demographics data
demogr_income = demogr.iloc[:, :500].copy()                                                    # Select only Income as demographics
demogr = demogr_income
# Drop outside good
df = df[df.nest != 'OUTSIDE']
df.reset_index(drop = True, inplace = True)                                      #Nest is used to have less genres

#--------------------------------------------------------------------------
#Embed dummies in the df: dummies for genre are required onli in BLP2 (Isac)
#--------------------------------------------------------------------------
#nest_dummies = pd.get_dummies(df['nest'], prefix='genre')                       # Missing line in th eorignal code (Isac)
#df = pd.concat([df, nest_dummies], axis=1)

# Unique ID for each market
df['mkt_id'] = df.groupby(['year','country'], sort = False).ngroup().astype(int)                # Unique ID for each prod country combination
cdid_demogr = list(np.array(df.mkt_id))                                                         # All the id's ordered are collected in cdid
cdid = [float(i) for i in cdid_demogr]                                                            
cdid = list(np.array(cdid))

# Calculate cdindex as the cumulative sum of firms in each mkt
df['n_apps_mkt'] = df.groupby(['mkt_id'])['constant'].transform('sum')              # cdindex is the last firm id for each mkt
cdindex = df.groupby(['mkt_id']).count().n_apps_mkt                                 # calculated as the cumulative sum of the firms-1 (b/c first firm is id =0)
cdindex = cdindex.cumsum()
cdindex = np.array(list(cdindex))
cdindex = cdindex-1                                                                  


#%%
# All the data needed for the estimation
class data():

# Variables

    # #Set of product characteristics (price, average score, in app purchase)
    x1 = df.loc[:,['constant','price','averagescore','iap']]                                                      
    x1.columns = [0,1,2,3]
    x1.loc[:,[1,2,3]] = x1.loc[:,[1,2,3]].div(scale)                                    #scale=100 set before. (Isac)

    #Allow for random coefficients on prices by interacting with demogrtaphics
    x2 = df.loc[:,['price']]
    x2 = x2.div(scale)
    x2 = np.array(x2)
    x2 = np.reshape(x2, (np.shape(x2)[0],1))

    #------------------------------------------------------------
    #Set of instruments - BLP1: Group_n as instrument (Isac)
    #------------------------------------------------------------
    IV = df.loc[:, ['group_n','constant', 'averagescore','iap']]
    IV.columns = [0,1,2,3]
    IV.iloc[:, [0,2,3]] = IV.iloc[:, [0,2,3]].div(scale)

    
    #Observed market shares of brand j in market t
    s_jt = np.array(df.s_jt)
    s_jt= np.reshape(s_jt, (np.shape(s_jt)[0],1))

    #Simulated individuals randomly and iid drawn from a normal distribution
    v = v                                                                       #It simply show previous simulation

    #Index of the last observation for a market
    cdindex = cdindex

    #Vector of market indexes
    cdid = cdid

    #Vector assigning market to each demographic observation
    cdid_demogr = cdid_demogr

    
    demogr = demogr.div(1000)

    #Number of simulated indivuduals
    ns = ns

# =============================================================================
# BLP Algorithm
#
#
# We will use the Nelder-Mead algorithm to find points in which the optimal value of an objective function in achieved.
# N.B: Many optimizers could be used to solve non-linear problems and Dub́e et al (2012) and Knittel and Metaxoglou (2014) 
#      have highlighted that other derivative-based routines perform better than the Nelder-Mead algorithm. Just for the 
#      sake of simplicity we will use the simple Nelder-Mead algorithm, using an appropriate tollerance for the inversion and
#      assuming we are using proper instruments. 
#
#
# OUTPUTS:
# 
# csv file with:
#   theta1: vector of estimated parameters for the indiriect utility function 
#   theta2: matrix of estimated interactions with demographic parameters for
#          the indiriect utility function
#   se: matrix of standard errors
#
# histogram:
#   alfa_i_reshaped: estimated distribution of the price coefficient (interacted with the demographics)
#                
# =============================================================================

from scipy.optimize import minimize
import matplotlib.pyplot as plt    
import time
    
class BLP:
    
    def __init__ (self,data,theta2w,mtol,niter): #initialization of the class BLP
         
        self.niter = niter 
        self.mtol=mtol
        self.theta2w=theta2w 
        self.x1 = data.x1
        self.x2 = data.x2 
        self.s_jt = data.s_jt
        self.v = data.v
        self.cdindex = data.cdindex 
        self.cdid = data.cdid
        self.cdid_demogr=data.cdid_demogr
        self.vfull = data.v.loc[self.cdid_demogr,:] 
        dfull = data.demogr.loc[self.cdid_demogr,:]
        dfull.columns= pd.RangeIndex(len(dfull.columns))
        dfull.index = pd.RangeIndex(len(dfull.index))
        self.dfull = dfull
        self.demogr= data.demogr
        self.IV = data.IV
        self.K1 = self.x1.shape[1]
        self.K2 = self.x2.shape[1]
        self.ns = data.ns          
        self.D = int(self.dfull.shape[1]/self.ns)       
        self.T = np.asarray(self.cdindex).shape[0]        
        self.TJ = self.x1.shape[0]  
        self.J = int(self.TJ/self.T) 
        
        
        self.invA = np.linalg.inv(self.IV.T@self.IV)
        
        
        # Calculate the outside good shares (s0)
        temp= self.s_jt.cumsum()                                                                # This makes the cumulative sum of all mkts shares in all mkts
        sum1 = temp[self.cdindex]                                                               # Cumulative sum of last firm in each mkt
        sum1[1:np.shape(sum1)[0]] = np.diff(sum1)                                               # Calculates the share of all firms in each mkt
        outshr = (1 - sum1[np.asarray(self.cdid,dtype=int)])                                    # Calculates outside share as 1- cumulative share
        outshr= np.abs(outshr)                                                                  # Outside share in abs value
        outshr=np.reshape(outshr,(np.shape(outshr)[0],1))                                       # Reshape

        #To have an educated guess of the coefficients, we run a simple logit regression
        y=np.log(self.s_jt) - np.log(outshr)
        mid = self.x1.T@self.IV@self.invA@self.IV.T
        t = np.linalg.inv(mid@self.x1)@mid@y 
        
        #Initial Mean Utility
        self.mvalold = self.x1@(t)
        self.mvalold = np.exp(self.mvalold).values.reshape(np.shape(self.mvalold)[0],1)
        self.oldt2 = np.zeros(np.shape(self.theta2w)) 
        self.gmmvalold = 0
        self.gmmdiff = 1
        self.gmmresid = np.ones((16942,1)) 
   
    #We want to transform the guessed theta2w into an ndarray
    def init_theta(self,theta2w):
        theta2w=theta2w.reshape(self.K2,1+self.D) 
        self.theti, self.thetj = list(np.where(theta2w != 0))
        self.theta2 = theta2w[np.where(theta2w != 0)]
        return self.theta2
     
        
    #Mean Utility Function
    def mufunc(self,theta2w): 
        mu = np.zeros((self.TJ, self.ns))
        for i in range(self.ns):
            v_i = np.array(self.vfull.loc[:, np.arange(i, self.K2*self.ns, self.ns)])
            d_i = np.array(self.dfull.loc[:, np.arange(i, self.D*self.ns, self.ns)]) #demographics that interact with price to from random coefficinets (Isac)
            temp = d_i @ theta2w[:, 1:(self.D+1)].T
            mu[:, i]=(np.multiply(self.x2, v_i) @ theta2w[:, 0]) + np.multiply(self.x2, temp) @ np.ones((self.K2))
        self.mu = mu
        return mu
        
        
    def ind_sh(self,expmu): 
        eg = np.multiply(expmu, np.kron(np.ones((1, self.ns)), self.mvalold)) 
        self.expmu = expmu
        self.eg = eg
        temp = np.cumsum(eg, 0)
        self.temp = temp
        sum1 = temp[self.cdindex, :]
        self.sum1 = sum1
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        self.sum2 = sum2
        denom1 = 1. / (1. + sum2)
        self.denom1 = denom1
        denom = denom1[np.asarray(self.cdid,dtype=int), :]
        self.denom = denom
        
        return np.multiply(eg, denom)

    def mktsh(self,expmu):
        
        #Compute the market share for each product
        temp = self.ind_sh(expmu).T
        f = (sum(temp) / float(self.ns)).T
        f = f.reshape(self.x1.shape[0],1)
        return f
        
    # Invert the shares to get the mean utility
    def meanval(self,theta2):

        if np.ndarray.max(np.absolute(theta2-self.oldt2)) < 0.01: 
            tol = self.mtol
            flag=0
        else:
            tol = self.mtol 
            flag = 1
        norm = 1
        avgnorm = 1
        i = 0
        theta2w = np.zeros((self.K2,self.D+1)) 
        for ind in range(len(self.theti)): 
            theta2w[self.theti[ind], self.thetj[ind]] = theta2[ind] 
        expmu=np.exp(self.mufunc(theta2w))
       
        while (norm > self.mtol) & (i<self.niter): 
           
            pred_s_jt = self.mktsh(expmu) 
            self.mval = np.multiply(self.mvalold,self.s_jt) / pred_s_jt 
            t = np.abs(self.mval - self.mvalold)
            norm = np.max(t)
            avgnorm = np.mean(t)
            self.mvalold = self.mval
            i += 1
           
            if (norm > self.mtol) & (i > self.niter-1):
                print('Max number of ' + str(niter) + 'iterations reached')
    
        print(['# of iterations for delta convergence: ' , i])
	
        if (flag == 1) & (sum(np.isnan(self.mval)))==0: 
            self.mvalold = self.mval
            self.oldt2 = theta2
        return np.log(self.mval)
 
 
    #Jacobian 
    def jacob(self,theta2): 
        cdindex=np.asarray(self.cdindex,dtype=int)
        theta2w = np.zeros((self.K2, self.D+1))
        for ind in range(len(self.theti)):
            theta2w[self.theti[ind], self.thetj[ind]] = self.theta2[ind] #Build theta2w (from array to ndarray) to plug in ind_sh
        expmu=np.exp(self.mufunc(theta2w)) 
        shares = self.ind_sh(expmu)
        f1 = np.zeros((np.asarray(self.cdid).shape[0] ,self.K2 * (self.D + 1)))
        
        #Calculate derivative  of shares with r (variable that are not interacted with demogr var, sigmas)
        for i in range(self.K2):
            xv = np.multiply(self.x2[:, i].reshape(self.TJ, 1) @ np.ones((1,self.ns)),self.v.loc[self.cdid, self.ns*i:self.ns * (i+1)-1])
            temp = np.cumsum(np.multiply(xv, shares), 0).values
            sum1 = temp[cdindex, :]
            sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
            f1[:,i] = np.mean((np.multiply(shares, xv - sum1[np.asarray(self.cdid,dtype=int),:])),1) #Mean over columns
           
        for j in range(self.D):
            d = self.demogr.loc[self.cdid,self.ns*(j)+1:self.ns*(j+1)] 
            temp1 = np.zeros((np.asarray(self.cdid).shape[0],self.K2))  #probably not so useful (Isac)
            for i in range(self.K2):
                xd = np.multiply(self.x2[:, i].reshape(self.TJ, 1) @ np.ones((1,self.ns)), d)
                temp = np.cumsum(np.multiply(xd, shares), 0).values
                sum1 = temp[cdindex, :]
                sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
                temp1[:, i] = np.mean((np.multiply(shares, xd-sum1[np.asarray(self.cdid,dtype=int), :])), 1)
            f1[:,self.K2 * (j + 1):self.K2 * (j + 2)] = temp1 

        self.rel = self.theti + self.thetj * (max(self.theti)+1) 
        f = np.zeros((np.shape(self.cdid)[0],self.rel.shape[0]))
        n = 0
        
        for i in range(np.shape(self.cdindex)[0]):
            temp = shares[n:(self.cdindex[i] + 1), :]
            H1 = temp @ temp.T
            H = (np.diag(np.array(sum(temp.T)).flatten())-H1) / self.ns
            f[n:(cdindex[i]+1),:] = np.linalg.inv(H) @ f1[n:(cdindex[i] + 1),self.rel]
            n = cdindex[i] + 1
        return f
       
    #GMM Objective Function       
    def gmmobj(self,theta2):
        print(theta2)
        delta = self.meanval(theta2)
        self.delta = delta
        self.theta2=theta2
        if max(np.isnan(delta)) == 1: 
            f = np.ndarray((1,1),buffer=np.array([1e+10,1e+10])) 
        else:
            temp1 = self.x1.T @ self.IV
            temp2 = delta.T@self.IV
            self.theta1 = np.linalg.inv(temp1@self.invA@temp1.T)@temp1@self.invA@temp2.T
            self.gmmresid = delta - self.x1@self.theta1
            temp1 = self.gmmresid.T@(self.IV) 
            f = temp1@self.invA@temp1.T
            f=f.values
            if np.shape(f) > (1,1):
                temp = self.jacob(theta2).T 
                df = 2*temp@self.IV@self.invA@self.IV.T@self.gmmresid #Error: why df and not f is the GMM objective? (Isac)
        print('fval:', f[0,0]) #Value of the GMM Objective Function
        
        #To be used in meanval to phase tolerance based on GMM value
        self.gmmvalnew = f[0,0]
        self.gmmdiff = np.abs(self.gmmvalold - self.gmmvalnew)
        self.gmmvalold = self.gmmvalnew
        self.gmmvalprint = f[0, 0]  # added to print GMM value in a dataframe (Isac)

        return (f[0,0])
       
               

    #Calculates the gradients of the objective function based on the Jacobian
    def gradobj(self,theta2):
        temp = self.jacob(theta2).T 
        df = 2*temp@self.IV@self.invA@self.IV.T@self.gmmresid 
        df=df.values 
        print('this is the gradient '+str(df))
    
        return df
        
    #Maximizes the objective function
    def iterate_optimization(self,opt_func, param_vec,jac,options ): 
        res = minimize(opt_func, param_vec,method='nelder-mead',options=options)
        return res
            
    
    #Compute the Variance-Covariance matrix   
    def varcov(self,theta2): 
        Z = self.IV.shape[1]
        temp = self.jacob(theta2)                                              
        a = np.concatenate((self.x1.values, temp), 1).T @ self.IV.values
        IVres = np.multiply(self.IV.values,self.gmmresid.values @ np.ones((1, Z)))
        b = IVres.T@(IVres)
        aAa=a @ np.asarray(self.invA) @ a.T
        if np.linalg.det(a @ np.asarray(self.invA) @ a.T)!= 0:
            #inv_aAa = np.linalg.inv(a @ np.asarray(self.invA) @ a.T)
            inv_aAa = np.linalg.pinv(a @ np.asarray(self.invA) @ a.T)           #Pseudo Inverse (Isac)

            #identity_matrix = np.eye(aAa.shape[0])                             #Solving linear system (Isac)
            #inv_aAa = np.linalg.solve(aAa, identity_matrix)
        else:
            print('Error: singular matrix in covariate function ; forced result') #if jacobian has row of zeros
            inv_aAa = np.linalg.lstsq(a @ np.asarray(self.invA) @ a.T)
        f = inv_aAa @ a @ np.asarray(self.invA) @ b @ np.asarray(self.invA) @ a.T @ inv_aAa
        return f    


    #Computes estimates of the coefficients and its standard errors; outputs a histogram and a csv file with estimates
    def results(self,theta2):
        self.theta1=self.theta1.values
        var = self.varcov(self.theta2)
        var_d = var.diagonal()
        print("varcov not regularized" + str(var))
        print("diagonal of the varcov not regularized" + str(var_d))
        # Regularize the covariance matrix to ensure positive semidefiniteness (isac)
        epsilon = 1e-6
        var_regularized = var + epsilon * np.identity(var.shape[0])
        var=var_regularized
        print("varcov regularized" + str(var))
        print("diagonal of the varcov regularized" + str(var_d))
        # Check the shape of the covariance matrix
        if var.shape[0] != var.shape[1]:
            print("Covariance matrix is not square.")
        var_d = var.diagonal()
        if np.isnan(var_d).any():
            print("Invalid values in the diagonal elements of the covariance matrix: nan.")
        if (var_d < 0).any():
            print("Invalid values in the diagonal elements of the covariance matrix: negative.")
        se = np.sqrt(var_d)
        t = se.shape[0] - self.theta2.shape[0]
        
        print('Object vcov dimensions: ' + str(np.shape(var)) )
        print('Object se dimensions: ' + str(np.shape(se)))
        
        print('Object theta2w dimensions:     ' + str(np.shape(self.theta2)))
        print('Object t dimensions:     ' + str(np.shape(t)))
        

    #Histogram for the price coefficient distribution
        alfa_i=[]
        alfa_i2=[]
        for i in range(0,self.T): 
            data_market=np.reshape(self.demogr.loc[i,0:self.ns*self.D].values,(self.ns,self.D))
            v_market=np.reshape(self.v.loc[i,0:self.ns-1].values,(self.ns,1)) 
            alfa_i2.extend(np.add(data_market@(self.theta2[1:2]).T, self.theta2[0]*v_market[:,0]))   #adjusted since we have only 2 coefficients in theta2
            alfa_i.extend(data_market@(self.theta2[1:2].T))
            
        alfa_i=(self.theta1[1]+alfa_i2)/100
        inelastic_n = np.sum(alfa_i >=0)
        inelastic_p = inelastic_n/len(alfa_i)
        inelastic_df = pd.DataFrame({
            'N of inelastic demands': [inelastic_n],
            '% of inelastic demands': [inelastic_p]
        })
        inelastic_df.to_csv('inelastic_data_BLP1_hp.csv', index=False)

        h=plt.figure()
        plt.hist(alfa_i,bins=25,range=(np.min(alfa_i),np.max(alfa_i)))
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of the price coefficient')
        h.savefig('elasticities_BLP1_hp.png')
        #Reshape the alfa parameter
        alfa_i_reshaped=np.reshape(alfa_i,(self.ns,len(self.cdindex))).T 
        alfa_i_reshaped=alfa_i_reshaped[np.asarray(self.cdid,dtype=int),:]
        
        #Convert theta2 into ndarray to plug in ind_sh
        theta = np.zeros((self.K2, self.D+1))
        for ind in range(len(self.theti)):
            theta[self.theti[ind], self.thetj[ind]] = self.theta2[ind] 
        ######
        expmu=np.exp(self.mufunc(theta))
        f = self.ind_sh(expmu)
        mval=(self.x1@self.theta1)
     
        #From ind_sh function: here mvalold = gmmresid +mval
        eg = np.multiply(np.exp(self.mufunc(theta)), np.kron(np.ones((1, self.ns)), (mval+self.gmmresid))) 
        temp = np.cumsum(eg, 0)
        sum1 = temp[self.cdindex, :]
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        denom1 = 1. / (1. + sum2)                                                  
        denom = denom1[np.asarray(self.cdid,dtype=int), :]
        f22= np.multiply(eg, denom)
        ######
        f2 = np.sum(f22.T,0)/self.ns
        f2 = f2.T
        error=self.s_jt-f2     

        #Table with parameters and final csv file
        self.theta1_results= pd.DataFrame({'Theta1':self.theta1.reshape(self.K1,),'Std.Error_theta1': se[0:-self.theta2.shape[0]]},columns=(['Theta1','Std.Error_theta1'])) 
        self.theta2_results= pd.DataFrame({'Theta2':self.theta2.reshape(self.theta2.shape[0],), 'Std.Error_theta2':se[-self.theta2.shape[0]:]},columns=(['Theta2','Std.Error_theta2']))
        self.theta1_results.to_csv("theta1_BLP1_hp.csv")
        self.theta2_results.to_csv("theta2_BLP1_hp.csv")

        gmmval_df = pd.DataFrame({"value of the gmm function": [self.gmmvalprint]})  #added to print GMM value in a dataframe (Isac)
        gmmval_df.to_csv('gmmValue_BLP1_hp.csv', index=False)



# =============================================================================
# Define parameters to input in the algorithm    
# =============================================================================

starttime = time.time()

#Maximum number of iterations for convergence of mval-mvalold contraction 
niter = 2500

#Initial guess for coefficients
theta2w = np.array([-0.5, 0.5])                                                           #Adjusted for only income in the demographics

#Tolerance level for iterations
mtol= 1e-5

#Set optimization options
options={'disp': None,'maxiter': 100,'xatol':0.0001,'fatol':0.0001}

#Get output
data_instance = data()
blp = BLP(data_instance,theta2w,mtol,niter)

init_theta = blp.init_theta(theta2w) 
res = blp.iterate_optimization(opt_func=blp.gmmobj,param_vec=init_theta,jac=blp.gradobj,options=options)

blp.results(res)

endtime=time.time()
run = endtime-starttime
print('running time: ' + str(endtime-starttime))
a=time.time()

