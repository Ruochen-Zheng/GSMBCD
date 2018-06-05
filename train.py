# -- coding: utf-8 --
from __future__ import division
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.16f}".format(x)})

import os
import pandas as pd
import random
from tqdm import tqdm
import time
from datasets import datasets
import loss_functions as losses
from scipy.io import savemat
from partition_rules import partition_rules
from selection_rules import VB_selection_rules
from selection_rules import FB_selection_rules
from update_rules import update_rules
from base import utils as ut





work = np.array([84,  220,  478,  558,  596,  753, 1103, 2009, 2044, 2301, 2410,
       2514, 2746, 3694, 4054, 4249, 4429, 4764, 5110, 5299, 5340, 5447,
       5680, 5899, 6254, 6256, 6412, 6518, 6538, 6587, 6770, 6796, 6848,
       6881, 6917, 6975, 7055, 7121, 7188, 7456, 8217, 8479, 8925, 9190,
       9583, 9681, 9690, 9692, 9793, 9811, 9992])

def train(dataset_name, loss_name, block_size, partition_rule, 
          selection_rule, 
          update_rule, n_iters, L1, L2, reset=0, optimal=None, 
          root="", logs_path="", datasets_path="",stdout_freq=1,time_cost=60,fixed_step=0.00001,stop_criterion='time'):
    
    fname = ("%s/%s_%s_%d_%s_%s_%s_%d_%d_%d.npy" % 
            (logs_path, dataset_name, loss_name, block_size, partition_rule, 
             selection_rule, update_rule, n_iters, L1, L2))

    

 

    
    np.random.seed(1)
    # load dataset
    dataset = datasets.load(dataset_name, path=datasets_path)
    A, b, args = dataset["A"], dataset["b"], dataset["args"]
    print 'A.shape:',A.shape
    print 'L1:',L1
    args.update({"L2":L2, "L1":L1, "block_size":block_size, 
                 "update_rule":update_rule})

    # loss function
    lossObject = losses.create_lossObject(loss_name, A, b, args)
    # Get partitions
    partition = partition_rules.get_partition(A, b, lossObject, block_size, p_rule=partition_rule)

    # Initialize x
    #x = np.zeros(lossObject.n_params)
    x = np.zeros(lossObject.n_params)
    #L_block=lossObject.Lb_func(x, A, b, None)
    #print 'x.shape:',x.shape
    history = []

    pbar = tqdm(desc="starting", total=n_iters, leave=True)

    rows=A.shape[0]
    cols=A.shape[1]
    ###### TRAINING STARTS HERE ############

    print 'update_rule:',update_rule
    #print 'block_size:',block_size
    print 'selection_rule:',selection_rule
    #index=range(10000)
    #random.shuffle(index)

    block = np.array([])
    t=time.time()

    gs_var_redu=True

    LL=np.max(A)**2

    partial_estimates=0



    if update_rule in ['stochastic_lb_double','stochastic_lb','stochastic_LS','stochastic_LS_double','Lb-NN_refined_sto','Lb-NN_refined_sto_double']:
        if update_rule in ['stochastic_lb_double','stochastic_LS_double','Lb-NN_refined_sto_double']:
            gs_var_redu=True
        #elif update_rule=='stochastic_lb':
        else:
            gs_var_redu=False
        ###
        ###循环，以次数停止
        '''
        for i in range((n_iters + 1)):
            if i>10:
                t_end=time.time()
                print t_end-t
                break
        '''        
        ###循环


        ###以时间停止

        i=0
        i_last=-1



        while True:

            if stop_criterion=='time':
                t_end=time.time()
                if t_end-t>time_cost:


                    #loss = lossObject.f_func(x, A, b)
                    #dis2opt = loss - OPTIMAL_LOSS[dataset_name + "_" + loss_name+"_"+str(L1)]
                    #history += [{"loss":loss, "iteration":i, "selected":block}]
                    #print dis2opt
                    break

            elif 'pe' in stop_criterion:
                nums=float(stop_criterion.split(':')[1])
                if partial_estimates/rows>=nums:

                    #loss = lossObject.f_func(x, A, b)
                    #dis2opt = loss - OPTIMAL_LOSS[dataset_name + "_" + loss_name+"_"+str(L1)]
                    #history += [{"loss":loss, "iteration":i, "selected":block}]
                    #print dis2opt
                    break

        ###时间
            x_outer=x.copy()
            miu=lossObject.g_func(x_outer, A, b, block=None)
            partial_estimates+=A.shape[0]

            '''
            x_tilde=x.copy()
            if i > 0:
                s = x_tilde - last_x_tilde
                y = miu - last_full_grad
                step_size = np.linalg.norm(s)**2 / np.dot(s, y) / 10
                #step_size = np.dot(s, y)/np.linalg.norm(s)**2 / 50
                #if i>1:
                 #   step_size=min(step_size,last_step_size)
                #print 'step_size:',step_size
                #L_block=1/step_size#*10
            last_step_size=step_size
            last_full_grad = miu
            last_x_tilde = x_tilde
            '''


            #if i%stdout_freq==0:
                #loss = lossObject.f_func(x, A, b)
                #dis2opt = loss - OPTIMAL_LOSS[dataset_name + "_" + loss_name+"_"+str(L1)]
                #history += [{"loss":loss, "iteration":i, "selected":block}]
                #stdout = ("%d - %s_%s_%s - dis2opt:%.16f - nz: %d/%d" % 
                         #(i, partition_rule, selection_rule, update_rule, dis2opt, (x!=0).sum(), x.size) )   
            #if i%1==0:
                #print (stdout)

                #print 'nn_numbers:',sum(x==0)                

            #if (i > 5 and (np.array_equal(work, np.where(x>1e-16)[0]))):
                #history[-1]["converged"] = dis2opt

            #if (i > 5 and (dis2opt == 0 or dis2opt < 1e-8)):
            #    break


            # Check increase
            #if (i > 0) and (loss > history[-2]["loss"] + 1e-6): 
            #    raise ValueError("loss value has increased...")


            for j in range(100):

                



                randrow=random.sample(range(rows),5)
                #A_=A[randrow]
                #b_=b[randrow]
                A_new=A[randrow]
                b_new=b[randrow]

                if gs_var_redu==True:

                    if partition is None:
                        if i_last!=i:
                            block,g,pvb, args = VB_selection_rules.select(selection_rule, x,'1', A_new, b_new, lossObject, args, iteration=i,x_outer=x_outer,miu=miu)
                            partial_estimates+=pvb

                        else:
                            block,g,pvb, args = VB_selection_rules.select(selection_rule, x,'2', A_new, b_new, lossObject, args, iteration=i,x_outer=x_outer,miu=miu)
                            partial_estimates+=pvb
                    else:
                        block, args = FB_selection_rules.select(selection_rule, x, A, b, lossObject, args, partition, iteration=i)

                else:
                    if partition is None:
                    
                        block,g,pvb, args = VB_selection_rules.select(selection_rule, x,'3', A, b, lossObject, args, iteration=i,x_outer=x_outer,miu=miu)
                        partial_estimates+=pvb
                    else:
                        block, args = FB_selection_rules.select(selection_rule, x, A, b, lossObject, args, partition, iteration=i)

                


                if update_rule in ['stochastic_lb','Lb-NN_refined_sto'] and selection_rule=='all':
                    L_block=1/fixed_step

                else:
                    L_block=lossObject.Lb_func(x, A, b, block,index=None)
                    #L_block=LL
                
                #L_block=0.01
                


                
                #print sum(x_last),sum(x)
             
                #print L_block
                # Update block
                x,pup, args = update_rules.update(update_rule, x, A_new, b_new, lossObject, args=args, block=block, iteration=i,miu=miu,x_outer=x_outer,\
                L_block=L_block,L1=L1,step_size=step_size,g=g,rows=rows)
                partial_estimates+=pup
                i_last=i
            ##以时间更新i
            i+=1
               


        loss = lossObject.f_func(x, A, b)
        print loss


    else:     

        ##以次数停止
        '''
        for i in range(n_iters + 1):
            if i>100:
                t_end=time.time()
                print t_end-t
                break     
        '''        
        ##次数


        ##以时间停止
        i=0
        while True:
            t_end=time.time()
            if t_end-t>time_cost:
                #loss = lossObject.f_func(x, A, b)
                #dis2opt = loss - OPTIMAL_LOSS[dataset_name + "_" + loss_name+"_"+str(L1)]
                #history += [{"loss":loss, "iteration":i, "selected":block}]
                #print dis2opt
                break

            elif 'pe' in stop_criterion:
                nums=float(stop_criterion.split(':')[1])
                if partial_estimates/rows>=nums:

                    #loss = lossObject.f_func(x, A, b)
                    #dis2opt = loss - OPTIMAL_LOSS[dataset_name + "_" + loss_name+"_"+str(L1)]
                    #history += [{"loss":loss, "iteration":i, "selected":block}]
                    #print dis2opt
                    break


        ##时间

            x_outer=0
            miu=0
            L_block=0 





            # Compute loss
            #if i%stdout_freq==0:
                #loss = lossObject.f_func(x, A, b)
                #dis2opt = loss - OPTIMAL_LOSS[dataset_name + "_" + loss_name+"_"+str(L1)]
                #history += [{"loss":loss, "iteration":i, "selected":block}]
                #print 'nn_numbers:',sum(x==0)

                # if i == 10:
                #     import ipdb; ipdb.set_trace()  # breakpoint c7301fd5 //

                #stdout = ("%d - %s_%s_%s - dis2opt:%.16f - nz: %d/%d" % 
                         #(i, partition_rule, selection_rule, update_rule, dis2opt, (x!=0).sum(), x.size) )   
            #pbar.set_description(stdout)
            #if i%1==0:
                #print(stdout)

            # # Check convergence
            #if (i > 5 and (np.array_equal(work, np.where(x>1e-16)[0]))):
                #history[-1]["converged"] = dis2opt

            #if (i > 5 and (dis2opt == 0 or dis2opt < 1e-8)):
            #    break


            # Check increase
            #if (i > 0) and (loss > history[-1]["loss"] + 1e-6): 
            #    raise ValueError("loss value has increased...")

            # Select block
            if partition is None:
                #block, args = VB_selection_rules.select(selection_rule, x, A, b, lossObject, args, iteration=i)
                block,g,pvb, args =VB_selection_rules.select(selection_rule, x,'3', A, b, lossObject, args, iteration=i,x_outer=x_outer,miu=miu)
                partial_estimates+=pvb
            else:
                block, args = FB_selection_rules.select(selection_rule, x, A, b, lossObject, args, partition, iteration=i)


            #print block
            # Update block
            x,pup, args = update_rules.update(update_rule, x, A, b, lossObject, args=args, block=block, iteration=i,miu=miu,x_outer=x_outer,\
                L_block=L_block,L1=L1,step_size=step_size,g=g,rows=rows)
            partial_estimates+=pup
            #print L_block
            #print 1/L_block
            #时间
            i+=1


        loss = lossObject.f_func(x, A, b)
        print loss



    
    #print x.shape
    #pbar.close()
    #ut.save_pkl(fname, history)



    #history = pd.DataFrame(history)
    
    #history["loss"] -= OPTIMAL_LOSS[dataset_name + "_" + loss_name+"_"+str(L1)]

    #if stop_criterion=='time':
    #    history['time']=np.linspace(0,time_cost,history.shape[0])
    #elif 'pe' in stop_criterion:
    #    nums=int(stop_criterion.split(':')[1])
    #    history['time']=np.linspace(0,nums,history.shape[0])
    
    #return history,x
'''
        else:           
            for i in range(n_iters + 1):
                # Compute loss
                loss = lossObject.f_func(x, A, b)
                dis2opt = loss - OPTIMAL_LOSS[dataset_name + "_" + loss_name]
                history += [{"loss":loss, "iteration":i, "selected":block}]


                # if i == 10:
                #     import ipdb; ipdb.set_trace()  # breakpoint c7301fd5 //

                stdout = ("%d - %s_%s_%s - dis2opt:%.16f - nz: %d/%d" % 
                         (i, partition_rule, selection_rule, update_rule, dis2opt, (x!=0).sum(), x.size) )   
                #pbar.set_description(stdout)
                print(stdout)

                # # Check convergence
                if (i > 5 and (np.array_equal(work, np.where(x>1e-16)[0]))):
                    history[-1]["converged"] = dis2opt

                if (i > 5 and (dis2opt == 0 or dis2opt < 1e-8)):
                    break


                # Check increase
                if (i > 0) and (loss > history[-1]["loss"] + 1e-6): 
                    raise ValueError("loss value has increased...")

                # Select block
                if partition is None:
                    block, args = VB_selection_rules.select(selection_rule, x, A, b, lossObject, args, iteration=i)

                else:
                    block, args = FB_selection_rules.select(selection_rule, x, A, b, lossObject, args, partition, iteration=i)

                # Update block
                x, args = update_rules.update(update_rule, x, A, b, lossObject, args=args, block=block, iteration=i)
'''






