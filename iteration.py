
    fname = ("%s/%s_%s_%d_%s_%s_%s_%d_%d_%d.npy" % 
            (logs_path, dataset_name, loss_name, block_size, partition_rule, 
             selection_rule, update_rule, n_iters, L1, L2))
   
    np.random.seed(1)
    # load dataset
    dataset = datasets.load(dataset_name, path=datasets_path)
    A, b, args = dataset["A"], dataset["b"], dataset["args"]
    print 'A.shape:',A.shape
    
    args.update({"L2":L2, "L1":L1, "block_size":block_size, 
                 "update_rule":update_rule})

    # loss function
    lossObject = losses.create_lossObject(loss_name, A, b, args)
    # get blocks
    partition = partition_rules.get_partition(A, b, lossObject, block_size, p_rule=partition_rule)
    # initialize x
    x = np.zeros(lossObject.n_params)
    zeros=np.zeros(lossObject.n_params)
    print 'x.shape:',x.shape
    history = []
    pbar = tqdm(desc="starting", total=n_iters, leave=True)
    rows=A.shape[0]
    cols=A.shape[1]
