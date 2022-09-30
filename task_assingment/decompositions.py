import helper_functions as hf
# TODO: need Node
# TODO: need sparse rm (bisimilarity code)


def get_best_decomposition(configs, file_name = 'best_decomp_agent_'): 
    '''
    takes configuration class and finds bd (best decomposition) by running a tree search 
    Returns:
        projected_best_file_names [file_name_0, file_name_1, ...]
        projected_best_rms [rm_0, rm_1, ... ]
    '''
    # Initialize tree 
    root = Node(name = 'root', future_events = configs.future_events, all_events= configs.all_events, knapsack = configs.forbidden_set) #forbidden set is the starting knapsack
    
    # Execute Tree search 
    bd = root.small_traverse(configs.num_agents, configs.rm, configs, enforced_set = configs.enforced_set)
    
    projected_best_file_names = []
    projected_best_rms = []

    full_events = set(configs.future_events)
    event_spaces, _ = hf.get_event_spaces_from_knapsack(full_events, bd[1][0]) 
    for i, es in enumerate(event_spaces):
        projected_rm = sparse.project_rm(es, configs.rm)

        # write new file 
        rm_file_name = file_name + str(i)
        projected_rm.write_rm_file(file_name = rm_file_name)

        # record file name and rms 
        projected_best_file_names.append(rm_file_name)
        projected_best_rms.append(projected_rm)

    return projected_best_file_names, projected_best_rms

def get_trivial_decomposition(configs, file_name = 'trivial_decomp_agent_'): # HACK fix this stuff
    '''
    Gets trivial decomposition by giving each agent the complete event set.

    Probably actually need to do this for each agent in the event that "forbidden" events are there
    Should also for good measure probably do a "bisimilarity check" since it is important to know that
    the forbidden events are not making the decomposition impossible 

    current return:
    [rm_type, ... , rm_type] repeated num_agents times  

    '''
    #need to get the correct subspaces given the the "required" forbidden events

    rm_file_list = []
    rm_list = []
    for a in range(configs.num_agents):
        event_space_a = set()
        for e in configs.rm.events: # should be all events 
            if (e, a) not in configs.forbidden_set:
                event_space_a.add(e)

        # now you have Sigma_a, the event space for agent a
        projected_rm = sparse.project_rm(event_space_a, configs.rm)
        rm_file_name = file_name + str(a)
        projected_rm.write_rm_file(file_name = rm_file_name)

        rm_list.append(projected_rm)
        rm_file_list.append(rm_file_name)

    # Now i have all my rm_file_lists and  my rm_lists
    # I say you should do a bisimilarity check rn... 
    rm_p = sparse.put_many_in_parallel(rm_list)

    #agent_event_spaces_dict = 
    #is_decomposable(configs.rm, rm_p, agent_event_spaces_dict, num_agents, enforced_set = None, forbidden_set = None, prints = False): #rm_p should not be an entry in this? yy
    #get_event_spaces_from_knapsack(configs.all_events, knapsack = configs.future_events)
    
    if sparse.is_bisimilar(configs.rm, rm_p):
        return rm_file_list, rm_list
    else:
        raise Exception(f" With forbidden assignments {configs.forbidden_set} this reward machine is not decomposible")
        

    trivial_projection =  sparse.project_rm(configs.rm.events, configs.rm) # HACK: project down with entire event set? Will this not just be the RM? haha

    return [trivial_projection]*configs.num_agents # Would probably be better to write an individual file for each agent's rm?  or could use the same haha

