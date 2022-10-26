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
import time

start_time = time.time()


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
    file_name = 'full_line_of_buttons_' + str(num_agents) + "_agents_" + str(num_buttons) + '_buttons.txt'
    rm.write_rm_file(file_location, file_name)
    return rm


num_agents = 4
num_buttons = 3

rm = get_lob_rm_harder(num_agents,num_buttons)

room_1_agents = [0, 1] #, 2] #3,  4]
room_2_agents = [2,3 ] # ,4]# [3,4] #[5, 6, 7, 8, 9]

room_1_buttons = [1, 3 ]# , 5] #, 7, 10]
room_2_buttons = [2] #, 4] #, 6, 8, 9]

# minimum levels: num_buttons * (_on) * num_agents = 25 for sure
# 2 * 3 agents in room one * 3 buttons in room one = 2 * 3 * 3 = 18
# 2 * 2 agents in room two * 2 buttons in room two = 2 * 2 * 2 = 8 ## 25 + 18 + 8 = 51

room_dict = {'room_1':{'agents': room_1_agents, 'buttons': room_1_buttons}, 'room_2':{'agents': room_2_agents, 'buttons': room_2_buttons}}


def get_faed_lob(room_dict, num_agents, num_buttons):
    forbidden_agent_event_dict = {}
    for k in range(num_agents):
        agent_forbidden_list = []
        for j in range(num_buttons):
            for i in range(num_agents):
                if i != k:
                    fe = str(i + 1) + 'b' + str(j+1)
                    le = str(i+1) + 'l' + str(j+1)
                    agent_forbidden_list.append(fe)
                    agent_forbidden_list.append(le)
        forbidden_agent_event_dict[k] = agent_forbidden_list

    for room, rd in room_dict.items():
        ag = rd['agents'] 
        bt = rd['buttons']
        for k in range(num_agents):
            agent_forbid_list = forbidden_agent_event_dict[k] 
            if k not in ag:
                for b in bt:
                    aa = str(k + 1) + 'b' + str(b)
                    ll = str(k + 1) + 'l' + str(b)
                    agent_forbid_list.append(aa)
                    agent_forbid_list.append(ll)
            forbidden_agent_event_dict[k] = agent_forbid_list

    return forbidden_agent_event_dict            


forbidden_agent_event_dict = get_faed_lob(room_dict, num_agents, num_buttons)
print(forbidden_agent_event_dict[0])
enforced_agent_event_dict = {0:{}, 1:{}, 2:{}, 3:{}}#, 4:{}} #, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}}
incompatible_pairs = []
include_all = True

weights = [1, .5, 0]

configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs= incompatible_pairs, include_all = include_all)

print(f" The tree search has {len(configs.future_events)} levels.")
print("Tree Order: ", configs.future_events)
print(" ")

root = Node(name = 'root', future_events = configs.future_events, all_events= configs.all_events, knapsack = configs.forbidden_set) #forbidden set is the starting knapsack

bd = root.traverse_last_minute_change(configs)

hf.print_results(configs, bd)
print(" --- %s seconds ---" % (time.time() - start_time))

while True:
    knapsack_id = input(" Which TA do you want? ")
    knapsack = bd[1][int(knapsack_id)]
    # do stuff to save files
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
    
    shared_events_dict  = {} # {agent# : shared events list} agent # start at 0
    for i, esi in agent_event_spaces_dict.items():
        my_shared_events = shared_events & set(esi)
        shared_events_dict[i] = list(my_shared_events)

    individual_events_dict = {} #{agent: set()}  
    for a, es in agent_event_spaces_dict.items():
        individual_events_dict[a] = set(es)

    print(f' Group shared events are:         {shared_events}.')
    print(f' Individual shared events are:    {shared_events_dict}.')
    print(f' Accident events are:            {acc_set}.')
    print(f" Task Assignment is:             {agent_event_spaces_dict}.")

    plist = []
    for i in range(num_agents):
        p = bs.project_rm(set(agent_event_spaces_dict[i]), strategic_rm)
    
    pm_par = bs.put_many_in_parallel(plist)

    print("Bisimilarity check: ", bs.is_bisimilar(strategic_rm, pm_par))
    aap_list = []
    for i in range(num_agents):
        aapi = bs.get_accident_avoidance_rm_less(p[i], acc_set, rm) 


    experiment_time = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
    new_rm_file_location = 'data/saved_reward_machines/line_of_buttons/'

    file_name_list = []
    for i in range(num_agents):
        aap = aap_list[i]
        rm_file_name = experiment_time + '_' + str(num_agents) + '_agents_' + str(num_buttons) + '_buttons_' + '_best_' + str(a) + '.txt'
        file_name_list.append(rm_file_name)
        aap.write_rm_file(new_rm_file_location, rm_file_name)

    response = input("Do you want to save another knapsack? y/n ")
    if response == 'n':
        break

    



