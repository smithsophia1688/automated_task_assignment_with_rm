from cmath import exp
from datetime import datetime
import pickle
import os

import reward_machines.sparse_reward_machine as sparse
from task_assignment.configurations import Configurations
import task_assignment.get_decompositions as gd

# thoughts
# check the assigned ta match what is happening with the simulations
# (compare hardcoded local rm to what my projected ones are)

num_times = 1 # Number of separate trials to run the algorithm for
num_agents = 3     


# EXPERIMENT 
experiment_name = 'crafting'
#experiment_name = 'buttons'
#experiment_name = 'rendezvous_small'
#experiment_name = 'rendezvous'
#experiment_name = 'centralized_rendezvous'
#experiment_name = 'ihrl_rendezvous'
#experiment_name = 'iql_rendezvous'
#experiment_name = 'ihrl_buttons'
#experiment_name = 'iql_buttons'


weights = [5, 0, 0]

#### Pick your decompositions to test ####
random = False
trivial = False # ONLY USE IF THERE ARE NO INCOMPATIBLE EVENTS. TOO HARD 
best = True
centralized = False


if experiment_name == 'buttons' and num_agents == 3:
    rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/buttons/team_buttons_copy.txt') # I would really like this to work haha
    
    # Decomposition we expect (used only for comparison in analysis), mostly for reference and posterity 
    expected_set = {0:['by', 'br', 'g'], 1: ['bg', 'a2br', 'a2lr', 'by', 'br'], 2: ['a3br', 'a3lr', 'br', 'bg']}

    # Set Restrictions 
    enforced_agent_event_dict  = {0:['by'], 1:[], 2:[]}        # {agent : [required  events]}
    forbidden_agent_event_dict = {0:['bg', 'a2br', 'a2lr', 'a3br', 'a3lr'], 1:['g'], 2:['g']}  # {agent : [forbidden events]}

    # Set agent utility of each event
    agent_utility_function = {0 : {'by': 0, 'bg': 0 , 'g':0 , 'a2br': 0 , 'a2lr': 0 , 'a3br' : 0, 'a3lr': 0, 'br': 0},
                              1 : {'by': 0, 'bg': 0 , 'g':0 , 'a2br': 1 , 'a2lr': 1 , 'a3br' : 0, 'a3lr': 0, 'br': 0},
                              2 : {'by': 0, 'bg': 0 , 'g':0 , 'a2br': 0 , 'a2lr': 0 , 'a3br' : 0, 'a3lr': 0, 'br': 0}}    # { agent : {event: utility} }

    incompatible_pairs = [('a2br', 'a3br'), ('a2lr', 'a3lr')]
    #incompatible_pairs = []


    # Build configurations class
    configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, agent_utility_function = agent_utility_function, weights = weights, incompatible_pairs= incompatible_pairs)
    
    experiment_time = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
    new_rm_file_location = 'data/saved_reward_machines/buttons/'
    
    if best:
        projected_file_names, projected_rms, agent_event_spaces_dict, knap_score = gd.get_best_decomposition(configs, new_rm_file_location, experiment_time)
        #print("task assignment results:")
        #print(agent_event_spaces_dict)
    
    if trivial:
        decomposition = gd.get_trivial_decomposition(configs, new_rm_file_location, experiment_time)
    
    if random:
        decomposition = gd.get_random_decomposition(configs, new_rm_file_location, experiment_time )

    print(" past ta" )
    import  experiments.config.buttons_config_with_task_assignment as bc 
    from experiments.dqprm import run_multi_agent_experiment

    #tester = bc.buttons_config_ta(num_times, num_agents, projected_file_names) # Get test object from config script # NEED TO CORRECT WHERE I get my Local RMs 
    
    #run_multi_agent_experiment(tester, num_agents, num_times, show_print = True)
    


elif experiment_name == 'crafting' and num_agents == 3:
    rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full.txt') 
    print("in crafting, have rm")
    print("rm events: ", rm.events)
    # Decomposition we expect (used only for comparison in analysis), mostly for reference and posterity 
    #expected_set = {0:['by', 'br', 'g'], 1: ['bg', 'a2br', 'a2lr', 'by', 'br'], 2: ['a3br', 'a3lr', 'br', 'bg']}

    # Set Restrictions 
    enforced_agent_event_dict  = {0:[], 1:[], 2:[]}        # {agent : [required  events]}
    forbidden_agent_event_dict = {0:['craft'], 1:['craft'], 2:[]}  # {agent : [forbidden events]}

    # Set agent utility of each event (SKIP FOR NOW)
    #agent_utility_function = {0 : {'by': 0, 'bg': 0 , 'g':0 , 'a2br': 0 , 'a2lr': 0 , 'a3br' : 0, 'a3lr': 0, 'br': 0},
    #                          1 : {'by': 0, 'bg': 0 , 'g':0 , 'a2br': 1 , 'a2lr': 1 , 'a3br' : 0, 'a3lr': 0, 'br': 0},
    #                          2 : {'by': 0, 'bg': 0 , 'g':0 , 'a2br': 0 , 'a2lr': 0 , 'a3br' : 0, 'a3lr': 0, 'br': 0}}    # { agent : {event: utility} }

    incompatible_pairs = [('a1pl', 'a2pl'), ('a1dl', 'a2dl')]

    configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs = incompatible_pairs)
    
    experiment_time = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
    new_rm_file_location = 'data/saved_reward_machines/buttons/'
    
    if best:
        projected_file_names, projected_rms, agent_event_spaces_dict, knap_score = gd.get_best_decomposition(configs, new_rm_file_location, experiment_time)
        #print("task assignment results:")
        #print(agent_event_spaces_dict)
    
    if trivial:
        decomposition = gd.get_trivial_decomposition(configs, new_rm_file_location, experiment_time)
    
    if random:
        decomposition = gd.get_random_decomposition(configs, new_rm_file_location, experiment_time )

    print("past crafting ta" )
    

elif experiment_name == 'rendezvous_small' and num_agents == 2:
    print("this far")
    rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/gridworld_many_agent_rendezvous/2_agent_rendezvous_rm_small.txt') # I would really like this to work haha

    print(rm.events)
    # Decomposition we expect (used only for comparison in analysis), mostly for reference and posterity 
    expected_set = {0:['l1', 'r1', 'r'], 1: ['l2', 'r2', 'r']}
    
    # Set Restrictions 
    enforced_agent_event_dict  = {0:['l1'], 1:['l2']}        # {agent : [required  events]}
    forbidden_agent_event_dict = {0:[], 1:[]}                # {agent : [forbidden events]}

    agent_utility_function = {0 : {'l1': 0, 'r1': 0 , 'l2':1 , 'r2': 0 , 'r': 0 },
                              1 : {'l1': 0, 'r1': 0 , 'l2':1 , 'r2': 0 , 'r': 0 }}    # { agent : {event: utility} }
    
    incompatible_pairs = [('l1', 'l2'), ('r1', 'r2')]
    configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, agent_utility_function = agent_utility_function, weights = weights, incompatible_pairs=incompatible_pairs)
    
    experiment_time = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
    new_rm_file_location = 'data/saved_reward_machines/rendezvous_small/'

    if best:
        best_decomposition = gd.get_best_decomposition(configs, new_rm_file_location, experiment_time )
        print('best files are', best_decomposition[0])

    if trivial:
        trivial_decomposition = gd.get_trivial_decomposition(configs, new_rm_file_location, experiment_time)
        print('trivial files are', trivial_decomposition[0])
  
    if random:
        random_decomposition = gd.get_random_decomposition(configs, new_rm_file_location, experiment_time )
        print('random files are', random_decomposition[0])

else: 
    print(f'The {experiment_name} experiment with {num_agents} agents is not defined yet!') # this is a silly print statement 

    # Work through how many "random" options you are looking at with these resetrictions on buttons ( are there only like 2? ) 1st walk through "trivial" 
    # Implement running example _ best _ trivial (?) _ random
    # Get learing environment set up (?) You need to do this asap
    # Figure out the "big" experiments. Think about what cyrus did. What would be interesting in learning context. 
    # 
    # 
    