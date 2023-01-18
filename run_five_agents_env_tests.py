from cmath import exp
from datetime import datetime

import pickle
import os
from re import T
import reward_machines.sparse_reward_machine as sparse
from task_assignment.configurations import Configurations
import task_assignment.get_decompositions as gd
import task_assignment.bisimilarity_check as bs
import task_assignment.helper_functions as hf
from task_assignment.tree_search import Node 
import  Environments.five_agents.five_agents_env as env
import Environments.five_agents.multiagent_five_agents_env as multi_env


###  TEAM REWARD MACHINE  ###
agent_id = 2

file = f'/Users/sophia/Documents/Research/automated_task_assignment_with_rm/data/saved_reward_machines/five_agent/five_agent_testing/2022_11_29-01.47.02_AM_trivial_{agent_id}.txt'
rm =  sparse.SparseRewardMachine(file = file) 




### INDIVIDUAL EVENT SPACES ###


agent_event_space = {2: ['b', 'a', 'e3'], 4: ['e5', 'b', 'a'], 0: ['e1', 'a', 'b'], 3: ['a', 'b', 'e4'], 1: ['b', 'e2', 'a']}
#agent_event_space = {3: ['b', 'e4'], 1: ['b', 'e2'], 0:[], 2:[], 4:[]}

es1 = agent_event_space[0]
es2 = agent_event_space[1]
es3 = agent_event_space[2]
es4 = agent_event_space[3]
es5 = agent_event_space[4]

es_list = [set(es1), set(es2), set(es3), set(es4), set(es5)]

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
print("very first", strategic_rm.U)

print("STRATEGIC RM")
print(strategic_rm)

p1 = bs.project_rm(set(agent_event_space[0]), strategic_rm)
p2 = bs.project_rm(set(agent_event_space[1]), strategic_rm)
p3 = bs.project_rm(set(agent_event_space[2]), strategic_rm)
p4 = bs.project_rm(set(agent_event_space[3]), strategic_rm)
p5 = bs.project_rm(set(agent_event_space[4]), strategic_rm)

ps = [p1, p2, p3, p4, p5]
p_dict = {0: p1, 1: p2, 2: p3, 3: p4, 4: p5}

par = bs.put_many_in_parallel(ps)

print("Bisim", bs.is_bisimilar(par,strategic_rm ))
pi = p_dict[agent_id]
aap = bs.get_accident_avoidance_rm_less_2(pi, acc_set, rm, check = False)

print("aap1 done")
print(aap)

### download accident reward machine ###
file_name = 'five_agent_testing_1.txt'
file_location = '/Users/sophia/Documents/Research/automated_task_assignment_with_rm/data/saved_reward_machines/five_agent/five_agent_testing/'

aap.write_rm_file(file_location , file_name)

### play game (have take file name) ### <- will need to change play() function 
#env.play( my_shared_events, individual_events, file_name)

env.play(shared_events_dict[agent_id], agent_event_space[agent_id], agent_id,  file_name = file_name)
