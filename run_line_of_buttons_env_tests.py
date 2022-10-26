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
import  Environments.line_of_buttons.line_of_buttons_env as env
import Environments.crafting.multiagent_crafting_env as multi_env



def get_lob_rm(num_agents, num_buttons):
    rm = sparse.SparseRewardMachine()

    num_states = (num_buttons * 2) + 1
    U = [x for x in range(num_states)]
    print(U)
    T = set()
    rm.u0 = 0
    # add transitions and reward function
    events = set()
    for u in range(num_states - 1):
        if u % 2 == 0:
            button_number = int( (u / 2) + 1)
            for i in range(num_agents):
                e = str(i + 1) + 'b' + str(button_number)
                l = str(i + 1) + 'l' + str(button_number)
                rm.add_transition_open(u, u+1, e, 0)
                rm.add_transition_open(u+1, u, l, 0)
                events.add(e)
                events.add(l)
        else:
            button_number = int((u-1) / 2 + 1)
            e = 'b' + str(button_number) + '_on'
            #l = 'b' + str(button_number) + '_off'

            if u+1 == num_states - 1: 
                reward = 1
                rm.add_transition_open(u, u+1, e, 1)
                T.add(u+1)
                events.add(e)
            else:
                reward = 0
                rm.add_transition_open(u, u+1, e, 0)
                #rm.add_transition_open(u+1, u, l, 0)
                events.add(e)
                #events.add(l)
                
    rm.events = events
    rm.T= T
    file_location = 'data/saved_reward_machines/line_of_buttons/'
    file_name = 'line_of_buttons_' + str(num_agents) + "_agents_" + str(num_buttons) + '_buttons.txt'
    rm.write_rm_file(file_location, file_name)
    return rm


def get_lob_rm_harder(num_agents, num_buttons):
    rm = sparse.SparseRewardMachine()

    num_states = (num_buttons * 2) + 1
    U = [x for x in range(num_states)]
    print(U)
    T = set()
    rm.u0 = 0
    
    # add transitions and reward function
    events = set()
    
    #T.add(u_bad)

    for u in range(num_states - 1):
        if u % 2 == 0:
            button_number = int( (u / 2) + 1) # 1, 2, ... num_buttons
            for i in range(num_agents):
                e = str(i + 1) + 'b' + str(button_number)
                l = str(i + 1) + 'l' + str(button_number)
                rm.add_transition_open(u, u+1, e, 0)
                rm.add_transition_open(u+1, u, l, 0)
                events.add(e)
                events.add(l)
            
            for x in range(num_buttons):
                bn = x + 1
                if bn > button_number:
                    for j in range(num_agents):
                        new_int =  str(j+1) + str(bn) + str(button_number)
                        u_bad = - int(new_int)
                        eearly = str(j + 1) + 'b' + str(bn)
                        rm.add_transition_open(u, u_bad, eearly, 0)
                
        else:
            button_number = int((u-1) / 2 + 1)
            e = 'b' + str(button_number) + '_on'
            #l = 'b' + str(button_number) + '_off'

            if u+1 == num_states - 1: 
                reward = 1
                rm.add_transition_open(u, u+1, e, 1)
                T.add(u+1)
                events.add(e)
            else:
                reward = 0
                rm.add_transition_open(u, u+1, e, 0)
                #rm.add_transition_open(u+1, u, l, 0)
                events.add(e)
                #events.add(l)
                
    rm.events = events
    rm.T= T
    file_location = 'data/saved_reward_machines/line_of_buttons/'
    file_name = 'line_of_buttons_' + str(num_agents) + "_agents_" + str(num_buttons) + '_buttons.txt'
    rm.write_rm_file(file_location, file_name)
    return rm



###  TEAM REWARD MACHINE  ###
#rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/line_of_buttons/line_of_buttons_5_agents_3_buttons.txt') 

rm = get_lob_rm_harder(3,3)



agent_id = 0
print(rm)

### INDIVIDUAL EVENT SPACES ###
es1= ['1b1','1l1', 'b1_on', '1b2', 'b2_on'] #['1b1', '1l1','b1_on', '1b2','b2_on', '1b3', 'b3_on']
es2 =  ['b1_on']
es3 = ['3b3', '3l3', 'b2_on', 'b3_on']

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
print("very first", strategic_rm.U)
#x = 'pause'

print("STRATEGIC RM")
print(strategic_rm)
print("before", strategic_rm.U)
#bs.remove_dead_transitions(strategic_rm)

print(strategic_rm)

print(strategic_rm.U)

p = bs.project_rm(set(agent_event_space[agent_id]), strategic_rm)
print(p)

p2 = bs.project_rm(set(agent_event_space[1]),strategic_rm)
p3 = bs.project_rm(set(agent_event_space[2]),strategic_rm)
print(p2)
print(p3)

par = bs.put_many_in_parallel([p, p2, p3])

print("Bisim", bs.is_bisimilar(par,strategic_rm ))

aap1 = bs.get_accident_avoidance_rm_less(p, acc_set, rm)
aap = bs.get_accident_avoidance_rm_less_2(p, acc_set, rm)
print(aap1)
print("aap1 done")
print(aap)



### download accident reward machine ###
file_name = 'crafting_rm_3.txt'
file_location = '/Users/sophia/Documents/Research/automated_task_assignment_with_rm/data/saved_reward_machines/crafting_task/crafting_env_testing/'
aap.write_rm_file(file_location , file_name)

print("PAST aap")
print(aap)
print("TERM", aap.T)

### play game (have take file name) ### <- will need to change play() function 


#env.play( my_shared_events, individual_events, file_name)


env.play(shared_events_dict[agent_id], agent_event_space[agent_id], agent_id,  file_name = file_name)



