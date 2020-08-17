import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
###We checked and print the arguments
import sys
print('Number of arguments:', str(len(sys.argv)), 'arguments.')
if len(sys.argv)!=8:
    print("""
    ### usage: python GCC.py [datafile] [threshold] [maxscale] [CEthreshold] [lp] [lq] [Nsteps]
    ### example: python GCC.py test.txt 0.5 2 1e-5 1 2 1000
    ###[datafile] The dataset you want to analyze using circular coordinates in .txt format. The cols of the txt file are dimensions/variables; the rows of the txt file are samples.
    ###[threshold] The threhold on persistence which we use to select those significant cocyles from all cocycles constructed from the Vietoris-Rips complex built upon the data.
    ###[CEthreshold] The threshold that we use to determine the constant edges. When the coordinate functions' values changed below this threshold, we consider it as a constant edge and plot it.
    ###[maxscal] The maximal scale at which we shall construct the Vietoris-Rips complex for circular coordinate computation.
    ###[lp] [lq] The generalized penalty function is in form of (1-lambda_parameter)*L^[lp]+lambda_parameter*L^[lq].
    ###[Nsteps] How many iterations you want to run in the tensorflow optimizer to obtain our circular coordinates?

    ###Functionality of this code.
    ####Part1: Construct the Vietoris-Rips complex built upon the data and associated persistence diagrams and barcodes.
    #####Scatter plot and associated persistence diagrams and barcodes, with significant topological features selected.
    ####Part2: Output the circular coordinates with different penalty functions.
    ####Part3: Output the embeddings with different penalty functions.""")
    sys.exit()
print('Argument List:', str(sys.argv),'\n')
filenam=sys.argv[1]
print('Data file:', filenam)

import dionysus
import scipy as sp
import numpy as np
#From Python_code/utils.py
def coboundary_1(vr, thr):
    D = [[],[]]
    data = []
    indexing = {}
    ix = [0]*2
    for s in vr:
        if s.dimension() != 1:
            continue
        elif s.data > thr:
            break
        indexing.setdefault(s.dimension(),{})
        indexing.setdefault(s.dimension()-1,{})
        if not s in indexing[s.dimension()]:
            indexing[s.dimension()][s] = ix[s.dimension()]
            ix[s.dimension()] += 1
        for dat, k in enumerate(s.boundary()): 
            if not k in indexing[s.dimension()-1]:
                indexing[k.dimension()][k] = k[0]
                ix[k.dimension()] += 1
            D[0].append(indexing[s.dimension()][s]) #rows
            D[1].append(indexing[k.dimension()][k]) #cols
            data.append(1. if dat % 2 == 0 else -1.)
    return sp.sparse.csr_matrix((data, (D[0], D[1]))), indexing


def optimizer_inputs(vr, bars, cocycle, init_z, prime):
    bdry,indexing = coboundary_1(vr,max(bar.death for bar in bars))
    n, m = bdry.shape # edges X nodes
    #-----------------
    l2_cocycle = [0]*len(init_z) #reorganize the coordinates so they fit with the coboundary indices
    for i, coeff in enumerate(init_z):
        l2_cocycle[i] = coeff
    l2_cocycle = np.array(l2_cocycle)
    #-----------------
    f = np.zeros((n,1)) # cocycle we need to smooth out, reorganize to fit coboundary
    for c2 in cocycle:
        if c2.element<(prime//2):
            f[indexing[1][vr[c2.index]]] += c2.element
        else:
            f[indexing[1][vr[c2.index]]] += c2.element-prime  
    return l2_cocycle,f,bdry

import numpy as np
from numpy import *
import dionysus
#Dionysus2 only.
#This code is composed in such way that it produces the whole thing in a single pdf file.
dataset = np.loadtxt(filenam)

import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
title_str='Circular Coordinates with Generalized Penalty Functions'
# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at the end of the block, even if an Exception occurs.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
filenam = os.path.splitext(filenam)[0]
pdfnam = filenam+'_output.pdf'
print('Output file:', pdfnam,'\n')

now = datetime.datetime.now()
print('Start Time(VR computation):',now.strftime("%Y-%m-%d %H:%M:%S"))
with PdfPages(pdfnam) as pdf:
    ##############################
    #Scatter plots for datapoints#
    fig = plt.figure(figsize=(5,5), dpi=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', 'datalim')
    plt.scatter(dataset.T[0,:],dataset.T[1,:],s=10, c='b')
    plt.axis('equal')
    plt.title('Scatter plot of data points')
    pdf.savefig(fig)
    #                            #
    ##############################
    ##############################
    #Compute Persistence Diagrams#
    prime = 23
    maxscale = float(sys.argv[3])
    #Choose the prime base for the coefficient field that we use to construct the persistence cohomology.
    threshold = float(sys.argv[2])
    print('Base coefficient field: Z/', prime ,'Z',sep='')
    print('Maximal scale:', float(sys.argv[3]))
    print('Persistence threshold for selecting significant cocyles:',threshold,'\n')

    vr = dionysus.fill_rips(dataset, 2, float(sys.argv[3]))
    #Vietoris-Rips complex
    cp = dionysus.cohomology_persistence(vr, prime, True)
    #Create the persistent cohomology based on the chosen parameters.
    dgms = dionysus.init_diagrams(cp, vr)
    #Calculate the persistent diagram using the designated coefficient field and complex.
    now = datetime.datetime.now()
    print('End Time (VR-computation):',now.strftime("%Y-%m-%d %H:%M:%S"))
    ###Plot the barcode and diagrams using matplotlib incarnation within Dionysus2. This mechanism is different in Dionysus.
    #Plots of persistence barcodes of Vietoris-Rips complex for dimension 0 and 1.
    fig = plt.figure(figsize=(5,5), dpi=100)
    plt.title('Persistence Barcode for dim 0')
    dionysus.plot.plot_bars(dgms[0], show=True)
    pdf.savefig(fig)
    fig = plt.figure(figsize=(5,5), dpi=100)
    plt.title('Persistence Barcode for dim 1')
    dionysus.plot.plot_bars(dgms[1], show=True)
    pdf.savefig(fig)
    plt.close('all')

    #Plots of persistence diagrams of Vietoris-Rips complex for dimension 0 and 1.
    fig = plt.figure(figsize=(5,5), dpi=100)
    plt.title('Persistence Diagram for dim 0')
    dionysus.plot.plot_diagram(dgms[0], show=True)
    pdf.savefig(fig)
    fig = plt.figure(figsize=(5,5), dpi=100)
    plt.title('Persistence Diagram for dim 1')
    dionysus.plot.plot_diagram(dgms[1], show=True)
    pdf.savefig(fig)
    plt.close('all')


    ######Select and highlight the features selected.
    bars = [bar for bar in dgms[1] if bar.death-bar.birth > threshold and bar.death-bar.birth < float(sys.argv[3])]
    #Choose cocycle that persist at least threshold we choose.
    cocycles = [cp.cocycle(bar.data) for bar in bars]
    print('\nSignificant features:')
    for B_Lt in bars:
        print(B_Lt)

    ####################
    #PersistenceBarcode#
    #Red highlight ***ALL*** cocyles that persist more than threshold value on barcode, when more than one cocyles have persisted over threshold values, this plots the first one.
    fig = plt.figure(figsize=(5,5), dpi=100)
    dionysus.plot.plot_bars(dgms[1], show=False)
    Lt1 = [[bar.birth,bar.death] for bar in dgms[1] if bar.death-bar.birth > threshold]
    #Lt1 stores the bars with persistence greater than the [threshold].
    Lt1_tmp = [[bar.birth,bar.death] for bar in dgms[1] if bar.death-bar.birth > 0]
    for Lt in Lt1:
        loc=0
        target=Lt
        for g in range(len(Lt1_tmp)):
            if Lt1_tmp[g][0] == target[0] and Lt1_tmp[g][1] == target[1]:
                loc=g
        #Searching correct term
        plt.plot([Lt[0],Lt[1]],[loc,loc],'r-')
        #print(Lt)

    plt.title('Selected cocycles on barcodes (red bars)')
    pdf.savefig(fig)
    plt.close('all')
    #                  #
    ####################
    ####################
    #PersistenceDiagram#
    #Red highlight ***ALL*** cocyles that persist more than threshold value on diagram.
    fig = plt.figure(figsize=(5,5), dpi=100)
    dionysus.plot.plot_diagram(dgms[1], show=False)
    Lt2 = [[point.birth,point.death] for point in dgms[1] if point.death-point.birth > threshold ]
    #Lt2 stores the (multi-)points with persistence greater than the [threshold].
    for Lt in Lt2:
        plt.plot(Lt[0],Lt[1],'ro')
    plt.title('Selected cocycles on diagram (red points)')
    plt.figure(figsize=(5,5), dpi=100)
    pdf.savefig(fig)
    plt.close('all')
    #                  #
    ####################

    #                            #
    ##############################
    ##############################
    #    Visualization of GCC    #
    overall_coords = np.zeros(dataset.shape[0], dtype = float)
    #from Python_code import utils
    toll = float(sys.argv[4])#tolerance for constant edges.
    print('\nConstant edges, with coordinates difference <',toll)
    print('Optimizer iteration numbers=',int(sys.argv[7]))

    lp=int(sys.argv[5])
    lq=int(sys.argv[6])
    now = datetime.datetime.now()
    print('Start Time (GCC computation):',now.strftime("%Y-%m-%d %H:%M:%S"))
    for lambda_parameter in [0,0.5,1]:
        embedding = []
        fig = plt.figure(figsize=(5,5), dpi=100)
        plt.text(0.3,0.5,'Analysis of Circular coordinates \n (mod {} - {}*L{} + {}*L{})'.format(prime,1-lambda_parameter,lp,lambda_parameter,lq),transform=plt.gca().transAxes)
        pdf.savefig(fig)
        plt.close('all')
        print('Penalty function =>',(1-lambda_parameter),'*L^',lp,"+",lambda_parameter,"*L^",lq,sep='')
        for g in range(len(cocycles)):
            chosen_cocycle = cocycles[g]
            chosen_bar = bars[g]
            vr_L2 = dionysus.Filtration([s for s in vr if s.data <= max([bar.birth for bar in bars])])
            coords = dionysus.smooth(vr_L2, chosen_cocycle, prime)
            l2_cocycle,f,bdry = optimizer_inputs(vr, bars, chosen_cocycle, coords, prime)
            l2_cocycle = l2_cocycle.reshape(-1, 1)
            ##It does not seem to work to have double invokes here...
            import tensorflow as tf
            B_mat = bdry.todense()
            #print(B_mat.shape)
            #l2_cocycle=np.zeros((B_mat.shape[1],1))
            z = tf.Variable(l2_cocycle, trainable=True)
            cost_z = (1-lambda_parameter)*tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - tf.matmul(B_mat,z) ),lp ) ), 1/lp) + lambda_parameter* tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - tf.matmul(B_mat,z) ),lq ) ), 1/lq)
            #Gradient Descedent Optimizer
            #Adams Optimizer
            opt_adams = tf.train.AdamOptimizer(1e-4).minimize(cost_z)
            #The latter is much better in terms of result
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(int(sys.argv[7])):#How many iterations you want to run?
                    sess.run(opt_adams)
                res_tf=sess.run([z,cost_z])
            res_tf=res_tf[0]
            overall_coords=overall_coords+res_tf.T[0,:]
            color = np.mod(res_tf.T[0,:],1)

            fig = plt.figure(figsize=(5,5), dpi=100)
            plt.scatter(dataset.T[0,:],dataset.T[1,:],s=10, c=color, cmap="hsv",zorder=10)
            plt.clim(0,1)
            plt.colorbar()
            plt.axis('equal')
            plt.title('Circular coordinates, \n {}-th cocyle (mod {} - {}*L{} + {}*L{})'.format(g+1,prime,1-lambda_parameter,lp,lambda_parameter,lq))
            edges_constant = []
            thr = chosen_bar.birth
            #####Constatn edges
            #Want to check constant edges in all edges that were there when the cycle was created
            for s in vr:
                if s.dimension() != 1:
                    continue
                elif s.data > thr:
                    break
                if abs(color[s[0]]-color[s[1]]) <= toll:
                    edges_constant.append([dataset[s[0],:],dataset[s[1],:]])
            edges_constant = np.array(edges_constant)
            pdf.savefig(fig)
            plt.close('all')
            #print('Loop End Time:',now.strftime("%Y-%m-%d %H:%M:%S"))

            fig = plt.figure(figsize=(5,5), dpi=100)
            if edges_constant.T!=[]:
              plt.plot(edges_constant.T[0,:],edges_constant.T[1,:], c='k', alpha=.5)
            plt.scatter(dataset.T[0,:],dataset.T[1,:],s=10, c=color, cmap="hsv",zorder=10)
            plt.clim(0,1)
            plt.colorbar()
            plt.axis('equal')
            plt.title('Circular coordinates/constant edges, \n {}-th cocyle (mod {} - {}*L{} + {}*L{})'.format(g+1,prime,1-lambda_parameter,lp,lambda_parameter,lq))
            pdf.savefig(fig)
            plt.close('all')
            color_filenam = filenam+'_CircularCoordinates_'+str(lambda_parameter)+'_'+str(g)+'.txt'
            np.savetxt(color_filenam,color)
            print('Penalty function =>',(1-lambda_parameter),'*L^',lp,"+",lambda_parameter,"*L^",lq,' Coordinates=>',color_filenam,sep='')

            fig = plt.figure(figsize=(5,5), dpi=100)
            angle = np.arctan(dataset.T[0,:]/dataset.T[1,:])
            plt.scatter(angle,color,s=10, c='b',zorder=10)
            plt.ylim([0,1])
            plt.xlim([-np.pi/2,np.pi/2])
            plt.title('Correlation plot against angle, \n {}-th cocyle (mod {} - {}*L{} + {}*L{})'.format(g+1,prime,1-lambda_parameter,lp,lambda_parameter,lq))
            pdf.savefig(fig)
            plt.close('all')
            embedding.extend([[sin(a) for a in 2*pi*color], [cos(a) for a in 2*pi*color]])

            fig = plt.figure(figsize=(5,5), dpi=100)
            dist2 = np.sqrt(np.power(dataset.T[0,:],2)+np.power(dataset.T[1,:],2))
            plt.scatter(dist2,color,s=10, c='b',zorder=10)
            plt.ylim([0,1])
            plt.xlim([0,maxscale])
            plt.title('Correlation plot aginst distance, \n {}-th cocyle (mod {} - {}*L{} + {}*L{})'.format(g+1,prime,1-lambda_parameter,lp,lambda_parameter,lq))
            pdf.savefig(fig)
            plt.close('all')
            

        emb_filenam = filenam+'_Embedding_'+str(lambda_parameter)+'.txt'
        np.savetxt(emb_filenam, np.array(embedding))
        print('Penalty function =>',(1-lambda_parameter),'*L^',lp,"+",lambda_parameter,"*L^",lq,' Embeddings=>',emb_filenam,sep='')
        #We plot the final circular coordinates with all co-cycles combined.
        overall_edges_constant = []
        overall_thr = float(sys.argv[2]) #For the combined coordinates, we choose the global threshold.
        for s in vr:
            if s.dimension() != 1:
                continue
            elif s.data > overall_thr:
                break
            if abs(overall_coords[s[0]]-overall_coords[s[1]]) <= toll:
                overall_edges_constant.append([dataset[s[0],:],dataset[s[1],:]])
        overall_edges_constant = np.array(overall_edges_constant)

        fig = plt.figure(figsize=(5,5), dpi=100)
        if overall_edges_constant.T!=[]:
          plt.plot(overall_edges_constant.T[0,:],overall_edges_constant.T[1,:], c='k', alpha=.5)
        plt.scatter(dataset.T[0,:],dataset.T[1,:],s=10, c=overall_coords, cmap="hsv",zorder=10)
        plt.clim(0,1)
        plt.colorbar()
        plt.axis('equal')
        plt.title('Combined circular coordinates/constant edges \n (mod {} - {}*L{} + {}*L{})'.format(prime,1-lambda_parameter,lp,lambda_parameter,lq))
        pdf.savefig(fig)
        plt.close('all')

        fig = plt.figure(figsize=(5,5), dpi=100)
        angle = np.arctan(dataset.T[0,:]/dataset.T[1,:])
        plt.scatter(angle,overall_coords,s=10, c='b',zorder=10)
        plt.ylim([0,1])
        plt.xlim([-np.pi/2,np.pi/2])
        plt.title('Correlation plot \n (mod {} - {}*L{} + {}*L{})'.format(prime,1-lambda_parameter,lp,lambda_parameter,lq))
        pdf.savefig(fig)
        plt.close('all')

        fig = plt.figure(figsize=(5,5), dpi=100)
        dist2 = np.sqrt(np.power(dataset.T[0,:],2)+np.power(dataset.T[1,:],2))
        plt.scatter(dist2,color,s=10, c='b',zorder=10)
        plt.ylim([0,1])
        plt.xlim([0,maxscale])
        plt.title('Correlation plot aginst distance, \n {}-th cocyle (mod {} - {}*L{} + {}*L{})'.format(g+1,prime,1-lambda_parameter,lp,lambda_parameter,lq))
        pdf.savefig(fig)
        plt.close('all')

    now = datetime.datetime.now()
    print('End Time (GCC computation):',now.strftime("%Y-%m-%d %H:%M:%S"))
    #                            #
    ##############################
    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = filenam
    d['Author'] = 'HengruiLuo, AlicePatania, JisuKim, MikaelVejdemo-Johansson'
    d['Subject'] = 'Generalized Penalty for Circular Coordinate Representation'
    d['Keywords'] = 'GCC'
    d['CreationDate'] = datetime.datetime.today()
    d['ModDate'] = datetime.datetime.today()
