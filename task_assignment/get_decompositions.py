import task_assignment.helper_functions as hf
from task_assignment.tree_search import Node 
import task_assignment.bisimilarity_check as bis

from itertools import chain, combinations
import random

# this code could go into bisimilarity_check.py but I want it in its own file


def get_best_decomposition(configs, file_location, file_name): 
    '''
    takes configuration class and finds bd (best decomposition) by running a tree search 
    Returns:
        projected_best_file_names [file_name_0, file_name_1, ...]
        projected_best_rms [rm_0, rm_1, ... ]
    '''
    # Initialize tree 
    root = Node(name = 'root', future_events = configs.future_events, all_events= configs.all_events, knapsack = configs.forbidden_set) #forbidden set is the starting knapsack
    
    # Execute Tree search 
    bd = root.small_traverse(configs)
    
    event_spaces, _ = hf.get_event_spaces_from_knapsack(configs.all_events, bd[1][0]) # arbitrarily taking first knapsack with best score. 
    
    projected_best_file_names = []
    projected_best_rms = []
    for i, es in enumerate(event_spaces):
        projected_rm = bis.project_rm(es, configs.rm)

        # write new file 
        rm_file_name = file_name + '_best_' + str(i) + '.txt'
        projected_rm.write_rm_file(file_location , rm_file_name)

        # record file name and rms 
        projected_best_file_names.append(rm_file_name)
        projected_best_rms.append(projected_rm)

    return projected_best_file_names, projected_best_rms

def get_trivial_decomposition(configs, file_location, file_name): # HACK fix this stuff
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
    proj_rm_list = []

    minimal_knapsack = configs.forbidden_set

    event_space , agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, minimal_knapsack)

    for a, es in agent_event_spaces_dict.items():
        # es is my Sigma_a
        # a is name of agent 
        projected_rm_a = bis.project_rm(set(es), configs.rm)
        print("post problem?")
        rm_file_name = file_name + '_trivial_' + str(a) + '.txt'
        projected_rm_a.write_rm_file(file_location, rm_file_name)
        
        proj_rm_list.append(projected_rm_a) # build to put in parallel
        rm_file_list.append(rm_file_name)


    # Now I have what I want to return

    # You should sitll probably do a decomposition check right now... 

    rm_p = bis.put_many_in_parallel(proj_rm_list)
    if bis.is_decomposable(configs.rm, rm_p, agent_event_spaces_dict, configs.num_agents, enforced_set = configs.enforced_set): 
        return rm_file_list, proj_rm_list
    else:
        raise Exception(f" With forbidden assignments {configs.forbidden_set} this reward machine is not decomposible (don't trust the trivial {file_name} output files)")
        
def get_random_decomposition(configs, file_location, file_name): # I am by far the least excited for this #This is also kind of dumb cause all it is saying is that an exhaustive search is better lol
    '''
    If this function works it will be a miracle
    '''
    possible_knapsacks_events = configs.all_events - configs.enforced_set
    knapsack_list = list(possible_knapsacks_events) #list of tuples 
    knapsack_powerset = list(chain.from_iterable(combinations(knapsack_list, r) for r in range(len(knapsack_list)+1))) #list of tuples where tuples are subsets of knapsack
    
    decomposition_success = False
    while not decomposition_success: 
        rm_file_list = []
        proj_rm_list = []

        random_knapsack_tuple = random.choice(knapsack_powerset)  #should be tuple of tuples
        random_knapsack = set(random_knapsack_tuple)

        # need to get all possible knapsacks now  

        if configs.forbidden_set.issubset(random_knapsack):
            event_space , agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, random_knapsack)

            for a, es in agent_event_spaces_dict.items():
                print("event set", es)
                projected_rm_a = bis.project_rm(set(es), configs.rm)
                proj_rm_list.append(projected_rm_a) # build to put in parallel
                
                rm_file_name = file_name + '_random_' + str(a) + '.txt'
                rm_file_list.append(rm_file_name)

            # Now I have what I want to return

            # You should sitll probably do a decomposition check right now... 
            rm_p = bis.put_many_in_parallel(proj_rm_list)

            if bis.is_decomposable(configs.rm, rm_p, agent_event_spaces_dict, configs.num_agents, enforced_set = configs.enforced_set): 
                decomposition_success = True

        else:
            knapsack_powerset.remove(random_knapsack_tuple)

    # If I made it here, rm_file_list and proj_rm_list are good to go, just write files
    for i, rm in enumerate(proj_rm_list):
        rm_file_name = rm_file_list[i]
        rm.write_rm_file(file_location, rm_file_name)

    return rm_file_list, proj_rm_list