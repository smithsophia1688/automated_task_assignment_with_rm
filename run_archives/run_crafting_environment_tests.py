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
import  Environments.crafting.crafting_env as env
import Environments.crafting.multiagent_crafting_env as multi_env



###  TEAM REWARD MACHINE  ###
rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full_no_cut.txt') 
agent_id = 0
print(rm)

### INDIVIDUAL EVENT SPACES ###
es1= ['a1', 'l1', 'timber', 'ar']
es2 = [ 'timber','tr2', 'ar']
es3 = [ 'l3', 'timber', 'a3', 'ar', 'craft']


agent_event_space = {0: es1, 1: es2, 2: es3}

es_list = [set(es1), set(es2), set(es3)]


### get accident set ###
strategy_set = set()
for es in es_list:
    strategy_set = strategy_set.union(es)

acc_set = rm.events.copy() - strategy_set
print(f'Strategy set is      {strategy_set}')
print(f'Accident events are  {acc_set}.')


### get shared event spaces ###
shared_events = set()
for e in strategy_set:
    share_count = 0
    for es in es_list:
        if e in es:
            share_count += 1 
    if share_count > 1:
        shared_events.add(e)
print(f'Shared events are    {shared_events}.')


shared_events_dict  = {} # {agent# : shared events list} agent # start at 0
for i, esi in enumerate(es_list):
    my_shared_events = shared_events & esi
    shared_events_dict[i] = list(my_shared_events)

print(f'my Shared events are    {shared_events_dict}.')

print(f'agent event dicts are ', agent_event_space)

### make accident reward machine ### 
strategic_rm = bs.get_strategy_rm(rm, strategy_set)
p = bs.project_rm(set(agent_event_space[agent_id]), strategic_rm)
aap = bs.get_accident_avoidance_rm(p, acc_set)

print(aap)

### download accident reward machine ###
file_name = 'crafting_rm_3.txt'
file_location = '/Users/sophia/Documents/Research/automated_task_assignment_with_rm/data/saved_reward_machines/crafting_task/crafting_env_testing/'
aap.write_rm_file(file_location , file_name)



### play game (have take file name) ### <- will need to change play() function 


#env.play( my_shared_events, individual_events, file_name)


env.play(shared_events_dict[agent_id], agent_event_space[agent_id], agent_id,  file_name = file_name)



