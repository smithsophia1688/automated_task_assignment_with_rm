from cmath import exp
from datetime import datetime
import pickle
import os

import reward_machines.sparse_reward_machine as sparse
from task_assignment.configurations import Configurations
import task_assignment.get_decompositions as gd
import task_assignment.bisimilarity_check as bs
import task_assignment.helper_functions as hf
from task_assignment.tree_search import Node 

### PROBABLY DO NOT NEED THE NEXT TWO.. ###
import  Environments.crafting.crafting_env as env
import Environments.crafting.multiagent_crafting_env as multi_env
import time

# trials = [[1, 0, 0], [0, 1, 0] , [0, 0, 1], [.5, .5, .5]]
trials = [[0, 0, 1]]
for xx in trials:
    start_time = time.time()

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
    ######################################           SET UP             #######################################
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#

    experiment_name = 'five'
    num_agents = 5
    weights = xx

    random = False
    trivial = False
    best = True
    centralized = False
    decomposition_types = [random, trivial, best, centralized]

    #rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full_no_cut.txt') # I would really like this to work haha
    rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/five_agent/five_agent_team_rm_early_consequences.txt') # I would really like this to work haha
    print(rm.events)
    forbidden_agent_event_dict = {0:['e2', 'e3', 'e4', 'e5'], 1:['e1', 'e3', 'e4', 'e5'], 2:['e2', 'e1', 'e4', 'e5'], 3: ['e1', 'e2', 'e3', 'e5'], 4: ['e1', 'e2', 'e3', 'e4']} 
    #enforced_agent_event_dict = {0: ['a1', 'l1', 'timber'], 1:['a2', 'l2', 'timber', 'tr2', 'ar'], 2: ['ar', 'craft']}
    
    agent_utility_function = {0:{'e1':1, 'e4':0, 'e5':0, 'b':0, 'a':0, 'e2':0, 'e3':0}, \
    1: {'e1':0, 'e4':0, 'e5':0, 'b':0, 'a':0, 'e2':4, 'e3':0}, \
        2:{'e1':0, 'e4':0, 'e5':0, 'b':0, 'a':0, 'e2':0, 'e3':2},\
            3: {'e1':0, 'e4':5, 'e5':0, 'b':0, 'a':0, 'e2':0, 'e3':0}, \
                4: {'e1':0, 'e4':0, 'e5':3, 'b':0, 'a':0, 'e2':0, 'e3':0}}


    enforced_agent_event_dict = {0:[], 1:[], 2:[]}
    incompatible_pairs = []
    include_all = False
    configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs= incompatible_pairs, include_all = include_all, agent_utility_function = agent_utility_function)


    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
    ######################################       TASK ASSIGNMENT        #######################################
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#


    root = Node(name = 'root', future_events = configs.future_events, all_events= configs.all_events, knapsack = configs.forbidden_set) #forbidden set is the starting knapsack
    print("tree events", configs.future_events)
    print(len(configs.future_events))
    print("ways: ", 2**len(configs.future_events))

    #bd = root.new_traverse(configs)

    bd = root.traverse_last_minute_change(configs)
    hf.print_results(configs, bd)
    #knapsack = bd[1][0] # arbitrary pick 

    print("--- %s seconds ---" % (time.time() - start_time))
    print(len(bd[1]))
    #knapsack_id = input(" Which TA do you want? ")
    
    knapsack_id = 0
    knapsack = bd[1][int(knapsack_id)]
    

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
    ######################################   PRE-LEARNING PROCESSING    #######################################
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#

    es_list , agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, knapsack)

    print("knapsack score ", bd[0])
    print("es list", es_list)
    
    # xxx = input("pause ")
    strategy_set = set()
    for es in es_list:
        strategy_set = strategy_set.union(es)

    acc_set = rm.events.copy() - strategy_set # Think (?) should this not actually be rm.events - each individual event set? (Update, it does matter don't do this)
    strategic_rm = bs.get_strategy_rm(rm, strategy_set)

    shared_events = set()
    for e in strategy_set:
        share_count = 0
        for es in es_list:
            if e in es:
                share_count += 1 
        if share_count > 1:
            shared_events.add(e)

    #print(f'Shared events are    {shared_events}.')

    shared_events_dict  = {} # {agent# : shared events list} agent # start at 0
    for i, esi in agent_event_spaces_dict.items():
        my_shared_events = shared_events & set(esi)
        shared_events_dict[i] = list(my_shared_events)

    #print(f'my Shared events are    {shared_events_dict}.')
    individual_events_dict = {} # {agent: set()}  
    for a, es in agent_event_spaces_dict.items():
        individual_events_dict[a] = set(es)

    print(f' Group shared events are:         {shared_events}.')
    print(f' Individual shared events are:    {shared_events_dict}.')
    print(f' Accident events are:            {acc_set}.')
    print(f" Task Assignment is:             {agent_event_spaces_dict}.")

    pms = []
    for i in range(num_agents):
        if i in agent_event_spaces_dict.keys():
            pi = bs.project_rm(set(agent_event_spaces_dict[i]), strategic_rm)
        else:
            pi = bs.project_rm(set(), strategic_rm)
            agent_event_spaces_dict[i] = set()
            shared_events_dict[i] = []
        pms.append(pi)

    pm_par = bs.put_many_in_parallel(pms)
    #print(pm_par)

    print(bs.is_bisimilar(strategic_rm, pm_par))
    aap_dict = {}
    for i in range(num_agents):
        pi = pms[i]
        aapi = bs.get_accident_avoidance_rm_less_2(pi, acc_set, rm)
        aap_dict[i] = aapi 


    experiment_time = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
    new_rm_file_location = 'data/saved_reward_machines/five_agent/'

    file_name_list = []

    for a in range(num_agents):
        aap = aap_dict[a]
        rm_file_name = experiment_time + '_best_' + str(a) + '.txt'
        file_name_list.append(rm_file_name)
        aap.write_rm_file(new_rm_file_location, rm_file_name)


    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
    ######################################        LEARNING SET UP       #######################################
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#

    # need to get learning moving still

    print("Now Past Task Assignment " )
    import  experiments.config.five_agents_config as cc

    from experiments.dqrm_five import run_multi_agent_experiment
    num_times = 10


    tester = cc.five_agents_config_ta(num_times, num_agents, file_name_list, agent_event_spaces_dict, shared_events_dict) # Get test object from config script # NEED TO CORRECT WHERE I get my Local RMs 
    tester.agent_event_spaces_dict = agent_event_spaces_dict
    tester.shared_events_dict = shared_events_dict

    # need to add shared events and individual events dicts for each agent 

    run_multi_agent_experiment(tester, num_agents, num_times, show_print = True)


    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    # sample usage


    new_rm_file_location = 'data/saved_testers/five_agent_task/'
    tester_file_name = experiment_time + f'_{weights[0]}_{weights[1]}_{weights[2]}.pkl'
    full_tester_file = new_rm_file_location + tester_file_name
    save_object(tester, full_tester_file)

    print("experiment time")
    print(experiment_time)
