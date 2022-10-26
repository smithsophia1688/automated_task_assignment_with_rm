from cmath import exp
from datetime import datetime
import reward_machines.sparse_reward_machine as sparse
from task_assignment.configurations import Configurations
import task_assignment.get_decompositions as gd
import task_assignment.bisimilarity_check as bs
import task_assignment.helper_functions as hf
from task_assignment.tree_search import Node 
import time

start_time = time.time()



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
                #rm.add_transition_open(u+1, u, l, 0)
                events.add(e)
                #events.add(l)
        else:
            button_number = int((u-1) / 2 + 1)
            e = 'b' + str(button_number) + '_on'
            l = 'b' + str(button_number) + '_off'

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
                
    rm.events = events
    rm.T= T
    file_location = 'data/saved_reward_machines/line_of_buttons/'
    file_name = 'line_of_buttons_' + str(num_agents) + "_agents_" + str(num_buttons) + '_buttons.txt'
    rm.write_rm_file(file_location, file_name)
    return rm



experiment_name = 'line_of_buttons'
num_agents = 5
num_buttons = 3
weights = [1, .5, 0]

rm = get_lob_rm(num_agents, num_buttons)
print(rm)


file_location = 'data/saved_reward_machines/line_of_buttons/'
file_name = 'line_of_buttons_' + str(num_agents) + "_agents_" + str(num_buttons) + '_buttons.txt'

rm.write_rm_file(file_location, file_name)

forbidden_agent_event_dict = {}
for k in range(num_agents):
    agent_forbidden_list = []
    for j in range(num_buttons):
        for i in range(num_agents):
            if i != k:
                fe = str(i + 1) + 'b' + str(j+1)
                agent_forbidden_list.append(fe)
    forbidden_agent_event_dict[k] = agent_forbidden_list


#print(forbidden_agent_event_dict)
enforced_agent_event_dict = {0:[], 1:[], 2:[]}
incompatible_pairs = []
include_all = True



configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs= incompatible_pairs, include_all = include_all)


print(f" The tree search has {len(configs.future_events)} levels.")
print("Tree Order: ", configs.future_events)
print(" ")

root = Node(name = 'root', future_events = configs.future_events, all_events= configs.all_events, knapsack = configs.forbidden_set) #forbidden set is the starting knapsack


bd = root.traverse_last_minute_change(configs)

#print("Levels: ", len(configs.future_events))
#print("  ")

bd = root.new_traverse(configs)
hf.print_results(configs, bd)

#knapsack = bd[1][0] # arbitrary pick 
#print("  ")
print(" --- %s seconds ---" % (time.time() - start_time))

