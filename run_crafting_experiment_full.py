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
start_time = time.time()



#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
######################################           SET UP             #######################################
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#

experiment_name = 'crafting'
num_agents = 3
weights = [1, .5, 0]

random = False
trivial = False
best = True
centralized = False
decomposition_types = [random, trivial, best, centralized]

#rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full_no_cut.txt') # I would really like this to work haha
rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full_no_cut.txt') # I would really like this to work haha
print(rm.events)
forbidden_agent_event_dict = {0:['a2', 'l2', 'a3', 'l3', 'craft', 'tr2', 'tr3'], 1:['a1', 'l1', 'a3', 'l3', 'craft', 'tr1', 'tr3'], 2:['a1', 'l1', 'a2', 'l2', 'tr1', 'tr2']} 
#enforced_agent_event_dict = {0: ['a1', 'l1', 'timber'], 1:['a2', 'l2', 'timber', 'tr2', 'ar'], 2: ['ar', 'craft']}

enforced_agent_event_dict = {0:[], 1:[], 2:[]}
#forbidden_agent_event_dict = {0:['craft'], 1:['craft'], 2:[]}
incompatible_pairs = []
configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs= incompatible_pairs)


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
#knapsack_id = input(" Which TA do you want? ")
knapsack_id = 0
knapsack = bd[1][int(knapsack_id)]

#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
######################################   PRE-LEARNING PROCESSING    #######################################
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#

es_list , agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, knapsack)

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
individual_events_dict = {} #{agent: set()}  
for a, es in agent_event_spaces_dict.items():
    individual_events_dict[a] = set(es)

print(f' Group shared events are:         {shared_events}.')
print(f' Individual shared events are:    {shared_events_dict}.')
print(f' Accident events are:            {acc_set}.')
print(f" Task Assignment is:             {agent_event_spaces_dict}.")

p1 = bs.project_rm(set(agent_event_spaces_dict[0]), strategic_rm)
p2 = bs.project_rm(set(agent_event_spaces_dict[1]), strategic_rm)
p3 = bs.project_rm(set(agent_event_spaces_dict[2]), strategic_rm)

pms = [p1, p2, p3]
pm_par = bs.put_many_in_parallel(pms)
print(pm_par)


print(bs.is_bisimilar(strategic_rm, pm_par))


aap1 = bs.get_accident_avoidance_rm_less(p1, acc_set, rm) 
aap2 = bs.get_accident_avoidance_rm_less(p2, acc_set, rm)
aap3 = bs.get_accident_avoidance_rm_less(p3, acc_set, rm)

aap_dict = {0: aap1, 1: aap2, 2: aap3}


experiment_time = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
new_rm_file_location = 'data/saved_reward_machines/crafting_task/'

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
import  experiments.config.crafting_config as cc

from experiments.dqprm_craft import run_multi_agent_experiment
num_times = 100


tester = cc.crafting_config_ta(num_times, num_agents, file_name_list, agent_event_spaces_dict, shared_events_dict) # Get test object from config script # NEED TO CORRECT WHERE I get my Local RMs 
tester.agent_event_spaces_dict = agent_event_spaces_dict
tester.shared_events_dict = shared_events_dict

# need to add shared events and individual events dicts for each agent 




run_multi_agent_experiment(tester, num_agents, num_times, show_print = True)


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
# sample usage


new_rm_file_location = 'data/saved_testers/crafting_task/'
tester_file_name = experiment_time + '_best.pkl'
full_tester_file = new_rm_file_location + tester_file_name
save_object(tester, full_tester_file)

print("experiment time")
print(experiment_time)
