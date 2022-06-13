"""
BAYESIAN NEURAL NETWORK WITH A SINGLE HIDDEN LAYER

This module trains a Bayesian neural network and then applies it to make predictions in the 
case of unknown and known targets.

Example:
    Initialize as bnn = CalcBNN()
    
Author:
    Payel Das
    Jason Sanders
    
To do:
    Nothing (I think).

"""

import numpy as np
import theano
TT = theano.tensor
import pymc3 as pm
#import cPickle as pickle # python 2
import pickle as pickle
import matplotlib.pyplot as plt

class CalcBNN3Layer:
    
    def  __init__(self):
        
        """ CLASS CONSTRUCTOR

        Arguments:
            None
        
        Returns:
            Nothing
        """        
      
        return
        
    def calcScale(self,data,eData):
        
        """ CALCULATES Z-SCORES FOR A SET OF DATA

        Arguments:
            data  - dataset (, [ndata*nvars], where ndata is the number of measurements and nvars is the number of variables)
            eData - observed errors on dataset (, [ndata*nvars])
        Returns:
            mu    - mean of each variable in dataset (, [nvars])
            sig   - standard deviation of each variable in dataset (, [nvars])
        """    
        
        mu  = np.mean(data,axis=0)
        sig = np.std(data,axis=0)
        
        return(mu,sig)

    def scaleData(self,data,eData,mu,sig):
        
        """ SCALES DATA ACCORDING TO SPECIFIED MEAN AND STANDARD DEVIATION

        Arguments:
            data  - dataset (, [ndata*nvars], where ndata is the number of measurements and nvars is the number of variables)
            eData - observed errors on dataset (, [ndata*nvars])
            mu    - mean of each variable in dataset (, [nvars])
            sig   - standard deviation of each variable in dataset (, [nvars])
        Returns:
            scaleData  - scaled dataset (, [ndata*nvars])
            scaleEData - scaled observed errors on dataset (, [ndata*nvars])
        """    

        scaleData  = (data-mu)/sig
        scaleEData = eData/sig

        return(scaleData,scaleEData)
    
    def unscaleData(self,scaledata,scaleeData,mu,sig):
        
        """ UNSCALES DATA ACCORDING TO SPECIFIED MEAN AND STANDARD DEVIATION

        Arguments:
            scaledata  - scaled dataset (, [ndata*nvars], where ndata is the number of measurements and nvars is the number of variables)
            scaleeData - scaled errors on dataset (, [ndata*nvars])
            mu         - mean of each variable in dataset (, [nvars])
            sig        - standard deviation of each variable in dataset (, [nvars])
        Returns:
            unscaleData  - unscaled dataset (, [ndata*nvars])
            unscaleEData - unscaled observed errors on dataset (, [ndata*nvars])
        """    

        unscaleData  = (scaledata*sig)+mu
        unscaleEData = scaleeData*sig

        return(unscaleData,unscaleEData)
        
    def trainBNN(self,inputsTrain,errInputsTrain,
                 targetsTrain,errTargetsTrain,
                 neuronsPerHiddenlayer,sampler,
                 nsamp,bnnmodelpkl,plotdir,
                 ncores=2,viewBNN=False):
        
        """ TRAINS BAYESIAN NEURAL NETWORK ACCORDING TO SPECIFIED TRAINING DATA, SAVES MODEL,
        AND VISUALIZES BNN IF DESIRED

        Arguments:
            inputsTrain           - input training set (, [ntrain*ninputs], where ntrain is the number of training measurements and ninputs is the number of inputs)
            errInputsTrain        - errors on input training set (, [ntrain*ninputs]) 
            targetsTrain          - target training set (, [ntrain*ntargets], where ntargets is the number of targets)
            errTargetsTrain       - errors on target training set (, [ntrain*ninputs]) 
            neuronsPerHiddenlayer - number of neurons in hidden layer
            sampler               - ADVI variational inference sampler or No U-Turn Sampler (NUTS) (much slower)
            nsamp                 - number of samples to generate
            bnnmodelpkl           - name of pickle file to store trained BNN
            plotdir               - directory for storing any associated plots
            ncores                - number of cores to use for NUTS sampler (default 2)
            viewBNN               - whether to visualize and plot BNN (default False)
        Returns:
            scaleData  - scaled dataset (, [ndata*nvars])
            scaleEData - scaled observed errors on dataset (, [ndata*nvars])
        """    
                     
        ntrain,ninputs  = np.shape(inputsTrain)
        ntrain,ntargets = np.shape(targetsTrain)
        
        # Calculate and scale inputs and targets
        inputsMu,inputsSig   = self.calcScale(inputsTrain,errInputsTrain)
        targetsMu,targetsSig = self.calcScale(targetsTrain,errTargetsTrain)
        inputsTrainScale,errInputsTrainScale = \
            self.scaleData(inputsTrain,errInputsTrain,inputsMu,inputsSig)
        targetsTrainScale,errTargetsTrainScale = \
            self.scaleData(targetsTrain,errTargetsTrain,targetsMu,targetsSig)
                     
        # Initialize weights, biases on neurons, and true X and Y
        np.random.seed(30)
        initWtsInHid   = np.random.randn(ninputs,neuronsPerHiddenlayer)
        initBiasInHid  = np.random.randn(neuronsPerHiddenlayer)
        initWtsHidHid  = np.random.randn(neuronsPerHiddenlayer,neuronsPerHiddenlayer)
        initBiasHidHid = np.random.randn(neuronsPerHiddenlayer)
        initWtsHidOut  = np.random.randn(neuronsPerHiddenlayer,ntargets)
        initBiasHidOut = np.random.randn(ntargets)
        initX          = np.random.randn(ntrain,ninputs)

        # Specify neural network
        with pm.Model() as neural_network:
    
    
            # Priors on weights and biases from input to first hidden layer
            wtsInHid  = pm.Normal('wtsInHid',
                                  mu      = 0,
                                  sd      = 1,
                                  shape   = (ninputs,neuronsPerHiddenlayer),
                                  testval = initWtsInHid)
            biasInHid = pm.Normal('biasInHid',
                                  mu      = 0,
                                  sd      = 1,
                                  shape   = (neuronsPerHiddenlayer,),
                                  testval = initBiasInHid)
                                  
            # Priors on weights and biases from first hidden layer to second hidden layer
            wtsHidHid  = pm.Normal('wtsHidHid',
                                   mu      = 0,
                                   sd      = 1,
                                   shape   = (neuronsPerHiddenlayer,neuronsPerHiddenlayer),
                                   testval = initWtsHidHid)
            biasHidHid = pm.Normal('biasHidHid',
                                   mu      = 0,
                                   sd      = 1,
                                   shape   = (neuronsPerHiddenlayer,),
                                   testval = initBiasHidHid)
    
            # Priors on weights and biases from second hidden layer to output
            wtsHidOut  = pm.Normal('wtsHidOut',
                                   mu      = 0,
                                   sd      = 1,
                                   shape   = (neuronsPerHiddenlayer,ntargets),
                                   testval = initWtsHidOut)
            biasHidOut = pm.Normal('biasHidOut',
                                   mu      = 0,
                                   sd      = 1,
                                   shape   = (ntargets,),
                                   testval = initBiasHidOut)
            
            # Priors on true inputs (mean zeros assuming they have been scaled, and std 1 means similar to measured values)
            xTrue  = pm.Normal('xTrue',
                               mu      = 0,
                               sd      = 10,
                               shape   = (ntrain, ninputs),
                               testval = initX) 
    
            # Expected outcome
            actHid1 = TT.nnet.sigmoid(TT.dot(xTrue,wtsInHid)+biasInHid)
            actHid2 = TT.nnet.sigmoid(TT.dot(actHid1,wtsHidHid)+biasHidHid)
            actOut  = TT.dot(actHid2,wtsHidOut)+biasHidOut 
            yTrue   = pm.Deterministic('yTrue',actOut)
            
            # Likelihoods of observations (sampling distribution - fixed)
            xTrainObs = pm.Normal('xTrainObs',
                                   mu         = xTrue,
                                   sd         = errInputsTrainScale,
                                   observed   = inputsTrainScale,
                                   total_size = (ntrain,ninputs))
            yTrainObs = pm.Normal('yTrainObs',
                                   mu         = yTrue,
                                   sd         = errTargetsTrainScale,
                                   observed   = targetsTrainScale,
                                   total_size = (ntrain,ntargets))          
                               
        # Train BNN
        print("Training Bayesian neural network with...")
        with neural_network:
    
            if (sampler=="advi"):
                # Fit with ADVI sampler
                print("   ...the ADVI sampler...")
                s         = theano.shared(pm.floatX(1))
                inference = pm.ADVI(cost_part_grad_scale=s)
                ftt       = pm.fit(n=nsamp, method=inference)
                trace     = ftt.sample(nsamp)
                fig       = plt.figure(figsize=(6,4))
                plt.plot(-ftt.hist)
                plt.ylabel('ELBO')
                fig.savefig(plotdir+"advi_fitprogress.eps")
                
            else:
                # Fit with NUTS sampler
                print("... ...the NUTS sampler...")
                step  = pm.NUTS(target_accept=0.95)
                ntune = 1000
                trace = pm.sample(nsamp,random_seed=10,step=step,tune=ntune,cores=ncores)
            print("...done.")
            
        # Save BNN to file
        print("Saving BNN, trace, and scaling of inputs and outputs to "+bnnmodelpkl+"...")
        with open(bnnmodelpkl,"wb") as buff:
            pickle.dump({'inputsMu':inputsMu,\
                         'inputsSig':inputsSig,\
                         'targetsMu':targetsMu,\
                         'targetsSig':targetsSig,\
                         'model': neural_network,\
                         'neuronsPerHiddenlayer': neuronsPerHiddenlayer,\
                         'trace': trace}, buff)
        print("...done.")
        
        if (viewBNN==True):
            
            # View neural_network model
            neural_network
        
            # View the free random variables (i.e. the ones you are obtaining posteriors for!) in the model
            neural_network.free_RVs
        
            # If desired plot neural network
            fig,ax=plt.subplots(7,2,figsize=(16,6))
            pm.traceplot(trace,ax=ax)
            fig.savefig(plotdir+"neural_network.eps",format='eps',dpi=100,bbox_inches='tight')
            
        return
    
    def calcPostPredSamplesTest(self,bnnmodelpkl,inputsTest,errInputsTest,
                                targetsTest,errTargetsTest,nppc,makePlots=False,
                                plotlabels=None,plotdir=None):
        
         
        """ CALCULATES POSTERIOR PREDICTIVE SAMPLES WITH KNOWN TARGETS (TEST CASE)
        Arguments:
            bnnmodelpkl           - name of pickle file to store trained BNN
            inputsTest            - input testing set (, [ntest*ninputs], where ntest is the number of testing measurements and ninputs is the number of inputs)
            errInputsTest         - errors on input testing set (, [ntest*ninputs]) 
            targetsTest           - target testing set (, [ntest*ntargets], where ntargets is the number of targets)
            errTargetsTest        - errors on target testing set (, [ntest*ninputs]) 
            nppc                  - number of posterior predictive samples
            makePlots             - whether to create plots
            plotlabels            - list of string for labelling target plots
            plotdir               - directory for storing any associated plots
        Returns:
            BNNData               - dataframe with mean inputs, standard deviation of inputs, mean of targets, and standard deviation oftargets (, [ntest*nvars])
            BNNFit                - dataframe with chi-squared, Bayesian p-value for the mean, and Bayesian p-value for the standard deviation for all targets (, [nvars])
        """  
    
        ntest,ninputs  = np.shape(inputsTest)
        ntest,ntargets = np.shape(targetsTest)

        # Load model and trace
        with open(bnnmodelpkl, "rb") as buff:
            data = pickle.load(buff)  
        inputsMu,inputsSig,targetsMu,targetsSig,neural_network,neuronsPerHiddenlayer,trace = \
            data['inputsMu'],\
            data['inputsSig'],\
            data['targetsMu'], \
            data['targetsSig'],\
            data['model'], \
            data['neuronsPerHiddenlayer'],\
            data['trace']
        
        # Calculate and scale inputs and targets
        inputsTestScale,errInputsTestScale = \
            self.scaleData(inputsTest,errInputsTest,inputsMu,inputsSig)
        targetsTestScale,errTargetsTestScale = \
            self.scaleData(targetsTest,errTargetsTest,targetsMu,targetsSig)

        # Add new nodes to the neural network and sample posterior predictive samples
        with neural_network:
            
            # New inputs (won't change because no likelihood)
            xTestTrue    = pm.Normal('xTestTrue',
                                     mu=inputsTestScale,
                                     sd=errInputsTestScale,
                                     shape=inputsTestScale.shape,
                                     testval=inputsTestScale)
            
            # New targets
            actHid1Test = TT.nnet.sigmoid(TT.dot(xTestTrue,neural_network.wtsInHid)+\
                                          neural_network.biasInHid)
            actHid2Test = TT.nnet.sigmoid(TT.dot(actHid1Test,neural_network.wtsHidHid)+\
                                          neural_network.biasHidHid)
            actOutTest  = TT.dot(actHid2Test,neural_network.wtsHidOut)+\
                                 neural_network.biasHidOut    
            yTestTrue   = pm.Normal('yTestTrue',
                                    mu    = actOutTest,
                                    sd    = 1.e-06,
                                    shape = (ntest,ntargets))
        
            # New likelihood
            yTestObs = pm.Normal('yTestObs',
                                 mu         = yTestTrue,
                                 sd         = errTargetsTestScale,
                                 observed   = targetsTestScale,
                                 total_size = (ntest,ntargets))         
        
            ppsTest = pm.sample_ppc(trace,vars=[xTestTrue,yTestTrue,yTestObs],samples=nppc,random_seed=10)

        # Estimate means and standard deviations for individual measurements
        inputsTrueModelMu,inputsTrueModelSig   = self.unscaleData(np.mean(ppsTest['xTestTrue'],axis=0),np.std(ppsTest['xTestTrue'],axis=0),inputsMu,inputsSig)
        targetsTrueModelMu,targetsTrueModelSig = self.unscaleData(np.mean(ppsTest['yTestTrue'],axis=0),np.std(ppsTest['yTestTrue'],axis=0),targetsMu,targetsSig)
        
        # Estimate population target mean and standard deviations
        targetsTrueModelMuPop,targetsTrueModelSigPop = self.unscaleData(np.mean(ppsTest['yTestTrue'],axis=1),np.std(ppsTest['yTestTrue'],axis=1),targetsMu,targetsSig)

        # Calculate chi-squared, Bayesian p-value for the mean, and Bayesian p-value for standard deviation
        ndof   = ntest
        chi2   = np.zeros(ntargets)
        pMu    = np.zeros(ntargets)
        pSig   = np.zeros(ntargets)
        plt.rc('font',family='serif')
        for jtargets in range(ntargets):
            
            print("TARGET "+str(jtargets)+":")
                
            # Chi-squared
            chi2[jtargets] = np.sum((targetsTest[:,jtargets]-targetsTrueModelMu[:,jtargets])**2./\
                                    (errTargetsTest[:,jtargets]**2+targetsTrueModelSig[:,jtargets]**2.))/ndof
            print("Reduced chi-squared (only valid for testing set) = "+str(chi2[jtargets]))
            
            # Calculate Bayesian p-values for the mean and standard deviation               
            targetMuPop  = np.mean(targetsTest[:,jtargets])
            targetSigPop = np.std(targetsTest[:,jtargets])
            pMu[jtargets] = np.sum(targetsTrueModelMuPop[:,jtargets]>targetMuPop)/float(nppc)
            print("Mean p-value = "+str(pMu[jtargets]))
            pSig[jtargets] = np.sum(targetsTrueModelSigPop[:,jtargets]>targetSigPop)/float(nppc)
            print("Standard deviation p-value = "+str(pSig[jtargets]))
            
            if (makePlots):
                                
                fig = plt.figure(figsize=(12,5))    
                fig.subplots_adjust(hspace=0.2,wspace=0.35)
  
                # Plot model against targets
                plt.subplot(1,2,1) 
                targetmin = min(targetsTest[:,jtargets])-0.05*abs(min(targetsTest[:,jtargets]))
                targetmax = max(targetsTest[:,jtargets])+0.05*abs(max(targetsTest[:,jtargets]))
                plt.errorbar(targetsTest[:,jtargets],targetsTrueModelMu[:,jtargets],
                             xerr=errTargetsTest[:,jtargets],yerr=targetsTrueModelSig[:,jtargets],
                             fmt='o',color='gray',linewidth=0.5,mfc='white')
                plt.plot([targetmin,targetmax],[targetmin,targetmax],':k',linewidth=1)
                plt.xlabel('Target', fontsize=14)
                plt.ylabel('Model', fontsize=14)
                plt.title(plotlabels[jtargets],fontsize=16)
                plt.xticks(fontsize = 12) 
                plt.yticks(fontsize = 12) 
                plt.xlim([targetmin,targetmax])
                plt.ylim([targetmin,targetmax])
          
                # Plot model errors against target errors
                plt.subplot(1,2,2) 
                errTargetmin = 0.*min(targetsTrueModelSig[:,jtargets])-0.05*abs(min(targetsTrueModelSig[:,jtargets]))
                errTargetmax = max(targetsTrueModelSig[:,jtargets])+0.05*abs(max(targetsTrueModelSig[:,jtargets]))
                plt.errorbar(errTargetsTest[:,jtargets],targetsTrueModelSig[:,jtargets],
                             xerr=0.,yerr=0.,fmt='o',color='gray',linewidth=0.5,mfc='white')
                plt.plot([errTargetmin,errTargetmax],[errTargetmin,errTargetmax],':k',linewidth=1)
                plt.xlabel('Target error', fontsize=14)
                plt.ylabel('Model error', fontsize=14)
                plt.title(plotlabels[jtargets],fontsize=16)
                plt.xticks(fontsize = 12) 
                plt.yticks(fontsize = 12) 
                plt.xlim([errTargetmin,errTargetmax])
                plt.ylim([errTargetmin,errTargetmax])
                plotfile = plotdir+"/modelcomp_target"+str(jtargets)+".eps"
                fig.savefig(plotfile,format='eps',bbox_inches='tight')
    
                fig = plt.figure(figsize=(6,7))    
                fig.subplots_adjust(hspace=0.2,wspace=0.35)
  
                # Plot population means
                plt.subplot(2,1,1) 
                n,bins,patches = plt.hist(targetsTrueModelMuPop[:,jtargets])
                plt.plot([targetMuPop,targetMuPop],[0.,1.1*np.max(n)])
                plt.xlabel('Population mean', fontsize=14)
                plt.xlabel('Number', fontsize=14)
                plt.xticks(fontsize = 12) 
                plt.yticks(fontsize = 12) 
                
                # Plot population standard deviations
                plt.subplot(2,1,2) 
                n,bins,patches = plt.hist(targetsTrueModelSigPop[:,jtargets])
                plt.plot([targetSigPop,targetSigPop],[0.,1.1*np.max(n)])
                plt.xlabel('Population standard deviation', fontsize=14)
                plt.xlabel('Number', fontsize=14)
                plt.xticks(fontsize = 12) 
                plt.yticks(fontsize = 12) 
                plotfile = plotdir+"/bayesp_target"+str(jtargets)+".eps"
                fig.savefig(plotfile,format='eps',dpi=1000,bbox_inches='tight')
                
            print(" ")
           
        # Make dictionary of model dataue
        BNNData = {'InputsTrueMu':inputsTrueModelMu,'InputsTrueSig':inputsTrueModelSig,
                   'TargetsTrueMu':targetsTrueModelMu,'TargetsTrueSig':targetsTrueModelSig}

        # Make dictionary of goodness-of-fit
        BNNFit = {'chi2':chi2,'pMu':pMu,'pSig':pSig}
        
        return(BNNData,BNNFit)

    def calcPostPredSamples(self,bnnmodelpkl,inputsNew,errInputsNew,nppc):
        
        """ CALCULATES POSTERIOR PREDICTIVE SAMPLES WITH UNKNOWN TARGETS 
        Arguments:
            bnnmodelpkl           - name of pickle file to store trained BNN
            inputsNew             - new input data (, [ndata*ninputs], where ndata is the number of measurements and ninputs is the number of inputs)
            errInputsNew          - errors on input testing set (, [ndata*ninputs]) 
            nppc                  - number of posterior predictive samples
        Returns:
            BNNData               - dataframe with mean inputs, standard deviation of inputs, mean of targets, and standard deviation oftargets (, [ntest*nvars])
            BNNFit                - dataframe with chi-squared, Bayesian p-value for the mean, and Bayesian p-value for the standard deviation for all targets (, [nvars])
        """    
        
        ndata,ninputs = np.shape(inputsNew)
        
        # Load model and trace
        with open(bnnmodelpkl, "rb") as buff:
            data = pickle.load(buff)  
        inputsMu,inputsSig,targetsMu,targetsSig,neural_network,\
        neuronsPerHiddenlayer,trace = \
            data['inputsMu'],\
            data['inputsSig'],\
            data['targetsMu'], \
            data['targetsSig'],\
            data['model'],\
            data['neuronsPerHiddenlayer'], \
            data['trace']
        
        # Calculate and scale inputs
        inputsNewScale,errInputsNewScale = \
            self.scaleData(inputsNew,errInputsNew,inputsMu,inputsSig)    
            
        # Determine number of targets
        ntargets = len(data["targetsMu"])
      
        # Add nodes for new data to neural network
        with neural_network:
            
            # New inputs (won't change because no likelihood)
            xNewTrue     = pm.Normal('xNewTrue',
                                     mu      = inputsNewScale,
                                     sd      = errInputsNewScale,
                                     shape   = inputsNewScale.shape,
                                     testval = inputsNewScale)
            
            # New target
            actHid1New = TT.nnet.sigmoid(TT.dot(xNewTrue,neural_network.wtsInHid)+\
                                        neural_network.biasInHid)
            actHid2New = TT.nnet.sigmoid(TT.dot(actHid1New,neural_network.wtsHidHid)+\
                                        neural_network.biasHidHid)
            actOutNew   = TT.dot(actHid2New,neural_network.wtsHidOut)+neural_network.biasHidOut    
            yNewTrue    = pm.Normal('yNewTrue',
                                    mu    = actOutNew,
                                    sd    = 1.e-6,
                                    shape = (ndata,ntargets)) 
    
            # Sample posterior predictive samples
            ppsNew = pm.sample_ppc(trace,vars=[xNewTrue,yNewTrue],samples=nppc,random_seed=20)
        
        # Estimate means and standard deviations for individual measurements
        inputsTrueModelMu,inputsTrueModelSig   = self.unscaleData(np.mean(ppsNew['xNewTrue'],axis=0),np.std(ppsNew['xNewTrue'],axis=0),inputsMu,inputsSig)
        targetsTrueModelMu,targetsTrueModelSig = self.unscaleData(np.mean(ppsNew['yNewTrue'],axis=0),np.std(ppsNew['yNewTrue'],axis=0),targetsMu,targetsSig)
                          
        # Make dictionary of model data
        BNNData = {'InputsTrueMu': inputsTrueModelMu, 
                   'InputsTrueSig': inputsTrueModelSig, 
                   'TargetsTrueMu': targetsTrueModelMu,
                   'TargetsTrueSig': targetsTrueModelSig}
         
        return(BNNData)