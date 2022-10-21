from email.mime import base
import sys
sys.path.append("..")

from tester.tester import Tester
from tester.tester_params import TestingParameters
from tester.learning_params import LearningParameters
import os

def crafting_config_ta(num_times, num_agents, rm_files, agent_event_spaces_dict, shared_events_dict):
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """
    base_file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    print("base_file_path")
    print(base_file_path)
    joint_rm_file = os.path.join(base_file_path, 'automated_task_assignment_with_rm', 'data', 'saved_reward_machines', 'crafting_task', 'crafting_rm_full_no_cut.txt')
    
    print("JOINT", joint_rm_file)
    local_rm_files = [] #add a config attribute to "what type of decomp" we are learningn on?

    for f in rm_files:
        full_file = base_file_path + '/automated_task_assignment_with_rm/data/saved_reward_machines/crafting_task/' + f
        local_rm_files.append(full_file )

    step_unit = 1000
    print("LOCAL FILES", local_rm_files)
    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1*step_unit 
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9 # 0.9
    learning_params.alpha = 0.8
    learning_params.T = 50
    learning_params.initial_epsilon = 0.0 # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)
    learning_params.max_timesteps_per_task = testing_params.num_steps

    tester = Tester(learning_params, testing_params)
    tester.step_unit = step_unit
    tester.total_steps = 250 * step_unit # 100 * step_unit
    tester.min_steps = 1

    tester.num_times = num_times
    tester.num_agents = num_agents

    tester.rm_test_file = joint_rm_file
    tester.rm_learning_file_list = local_rm_files
    tester.agent_event_spaces_dict = agent_event_spaces_dict
    tester.shared_events_dict = shared_events_dict

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 10
    env_settings['Nc'] = 10
    env_settings['initial_states'] = [20, 81, 99]
    env_settings['walls'] = [(0, 7), (1, 7), (2, 7), (3,7), (4,7), (5,7), (8,7), (9,7)]
    env_settings['craft_table'] = (1,9)
    env_settings['tree_loc'] = (0, 3)
    env_settings['log_loc'] = (0, 4)
    

    env_settings['p'] = 0.98

    tester.env_settings = env_settings

    tester.experiment = 'crafting'

    return tester