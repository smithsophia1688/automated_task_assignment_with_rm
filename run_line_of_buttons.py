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

experiment_name = 'line_of_buttons'
num_agents = 5
num_buttons = 5
weights = [1, .5, 0]

random = False
trivial = False
best = True
centralized = False
decomposition_types = [random, trivial, best, centralized]
rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/line_of_buttons/line_of_buttons_rm_full.txt') # I would really like this to work haha

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
#forbidden_agent_event_dict = {0:['2b1','2b1',]}
#enforced_agent_event_dict = {0: ['a1', 'l1', 'timber'], 1:['a2', 'l2', 'timber', 'tr2', 'ar'], 2: ['ar', 'craft']}
enforced_agent_event_dict = {0:[], 1:[], 2:[]}

incompatible_pairs = []
configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs= incompatible_pairs)

print(" ")
root = Node(name = 'root', future_events = configs.future_events, all_events= configs.all_events, knapsack = configs.forbidden_set) #forbidden set is the starting knapsack
print("Tree Order: ", configs.future_events)

print("Levels: ", len(configs.future_events))
print("  ")
bd = root.new_traverse(configs)
hf.print_results(configs, bd)

knapsack = bd[1][0] # arbitrary pick 
print("  ")
print(" --- %s seconds ---" % (time.time() - start_time))

