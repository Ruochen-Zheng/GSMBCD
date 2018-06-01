import pickle

with open('argfile','r') as f:
    data=pickle.load(f)

#data[0].u_rules=['stochastic_lb','Lb','stochastic_lb_double']
data[0].u_rules=['Lb-NN_refined_sto_double','Lb-NN_refined','Lb-NN_refined_sto']
data[0].s_rules=['GS','Random','all']

#data[0].u_rules=['Lb-NN_refined_sto']
#data[0].s_rules=['all']

data[0].loss_names=['lsl1nn_r']
#data[0].L1=[1e-2,1e-5,1e-9]
#data[0].L1=[0.01,0.00001,0.000000001]
data[0].dataset_names=['log1p.E2006.train']
data[0].L1=[1e2]

data[0].blockList=100
data[0].plot_names=['Lb-NN_refined_sto_double:GSMBCD2','Lb-NN_refined_sto:GSMBCD','Lb:GBCD']
data[0].stdout_freq=1
data[0].time_cost=300 #18
data[0].fixed_step=0.0001
data[0].stop_criterion='time'
#data[0].L2=10
#data[0].u_rules=['Lb-NN_refined_sto','Lb-NN_refined_sto_double','Lb-NN_refined']
