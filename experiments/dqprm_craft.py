import numpy as np
import random, time

from tester.tester import Tester
from Agent.agent import Agent
from Environments.coop_buttons.buttons_env import ButtonsEnv
from Environments.coop_buttons.multi_agent_buttons_env import MultiAgentButtonsEnv
from Environments.rendezvous.gridworld_env import GridWorldEnv
from Environments.rendezvous.multi_agent_gridworld_env import MultiAgentGridWorldEnv
from Environments.crafting.multiagent_crafting_env import MultiAgentCraftingEnv
from Environments.crafting.crafting_env import CraftingEnv
import matplotlib.pyplot as plt

def run_qlearning_task(epsilon,
                        tester,
                        agent_list,
                        show_print=True):
    """
    This code runs one q-learning episode. q-functions, and accumulated reward values of agents
    are updated accordingly. If the appropriate number of steps have elapsed, this function will
    additionally run a test episode.

    Parameters
    ----------
    epsilon : float
        Numerical value in (0,1) representing likelihood of choosing a random action.
    tester : Tester object
        Object containing necessary information for current experiment.
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    show_print : bool
        Optional flag indicating whether or not to print output statements to terminal.
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    num_agents = len(agent_list)

    for i in range(num_agents):
        agent_list[i].reset_state()
        agent_list[i].initialize_reward_machine()

    num_steps = learning_params.max_timesteps_per_task
    traject = {0:[], 1:[], 2:[]}
    # Load the appropriate environments for training
    if tester.experiment == 'crafting':
        training_environments = []
        for i in range(num_agents): 
            training_environments.append(CraftingEnv(agent_list[i].rm_file, i, tester.env_settings, tester.shared_events_dict[i], tester.agent_event_spaces_dict[i])) 

    for t in range(num_steps):
        # Update step count
        tester.add_step()

        for i in range(num_agents):
            # Perform a q-learning step.
            if not training_environments[i].is_complete():
                #if not(agent_list[i].is_task_complete): 
                current_u = agent_list[i].u
                s, a = agent_list[i].get_next_action(epsilon, learning_params)
                r, l, s_new = training_environments[i].environment_step(s,a)
                if l:
                    old_traj = traject[i] #TODO: PUT BREAK HERE
                    old_traj.append(l[0])
                    traject[i] = old_traj

                # a = training_environments[i].get_last_action() # due to MDP slip #TODO: there is slip
                agent_list[i].update_agent(s_new, a, r, l, learning_params) 


                '''
                for u in agent_list[i].rm.U:
                    if not (u == current_u) and not (u in agent_list[i].rm.T): # checking other, non terminal u
                        l = training_environments[i].get_mdp_label(s, s_new, u) 
                        r = 0
                        u_temp = u
                        u2 = u
                        for e in l:
                            # Get the new reward machine state and the reward of this step
                            u2 = agent_list[i].rm.get_next_state(u_temp, e)
                            r = r + agent_list[i].rm.get_reward(u_temp, u2)
                            
                            # Update the reward machine state
                            u_temp = u2
                        agent_list[i].update_q_function(s, s_new, u, u2, a, r, learning_params)
                '''

        # If enough steps have elapsed, test and save the performance of the agents.
        if testing_params.test and tester.get_current_step() % testing_params.test_freq == 0:
            t_init = time.time()
            step = tester.get_current_step()

            agent_list_copy = []

            # Need to create a copy of the agent for testing. If we pass the agent directly
            # mid-episode to the test function, the test will reset the world-state and reward machine 
            # state before the training episode has been completed.
            for i in range(num_agents):
                rm_file = agent_list[i].rm_file
                s_i = agent_list[i].s_i
                actions = agent_list[i].actions
                agent_id = agent_list[i].agent_id
                num_states = agent_list[i].num_states
                agent_copy = Agent(rm_file, s_i, num_states, actions, agent_id)
                # Pass only the q-function by reference so that the testing updates the original agent's q-function.
                agent_copy.q = agent_list[i].q.copy()
                agent_copy.my_shared_events = agent_list[i].my_shared_events.copy()
                agent_copy.individual_events = agent_list[i].individual_events.copy()
                
                agent_list_copy.append(agent_copy)

            # Run a test of the performance of the agents
            testing_reward, trajectory, testing_steps = run_multi_agent_qlearning_test(agent_list_copy,
                                                                                        tester,
                                                                                        learning_params,
                                                                                        testing_params,
                                                                                        show_print=show_print) #TODO: check 
            # Save the testing reward
            if 'reward' not in tester.results.keys():
                tester.results['reward'] = {}
            if step not in tester.results['reward']:
                tester.results['reward'][step] = []
            tester.results['reward'][step].append(testing_reward)
            



            # Save the testing trace
            if 'trajectories' not in tester.results.keys():
                tester.results['trajectories'] = {}
            if step not in tester.results['trajectories']:
                tester.results['trajectories'][step] = []
            tester.results['trajectories'][step].append(trajectory)

            # Save how many steps it took to complete the task
            if 'testing_steps' not in tester.results.keys():
                tester.results['testing_steps'] = {}
            if step not in tester.results['testing_steps']:
                tester.results['testing_steps'][step] = []
            tester.results['testing_steps'][step].append(testing_steps)

            # Keep track of the steps taken
            if len(tester.steps) == 0 or tester.steps[-1] < step:
                tester.steps.append(step)
        
        # If the agents has completed its task, reset it to its initial state.
        
        if all(env_i.is_complete() for env_i in training_environments):
            hhh = "all done with this step" #TODO: put break here
            for i in range(num_agents):
                agent_list[i].reset_state()
                agent_list[i].initialize_reward_machine() # re initialize the environement?
                training_environments[i] = training_environments[i].reset()

            # Make sure we've run at least the minimum number of training steps before breaking the loop
            if tester.stop_task(t):
                break

        # checking the steps time-out
        if tester.stop_learning():
            break
        

def run_multi_agent_qlearning_test(agent_list,
                                    tester,
                                    learning_params,
                                    testing_params,
                                    show_print=True):
    """
    Run a test of the q-learning with reward machine method with the current q-function. 

    Parameters
    ----------
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    learning_params : LearningParameters object
        Object storing parameters to be used in learning.
    Testing_params : TestingParameters object
        Object storing parameters to be used in testing.

    Ouputs
    ------
    testing_reard : float
        Reward achieved by agent during this test episode.
    trajectory : list
        List of dictionaries containing information on current step of test.
    step : int
        Number of testing steps required to complete the task.
    """
    num_agents = len(agent_list)

    if tester.experiment == 'rendezvous':
        testing_env = MultiAgentGridWorldEnv(tester.rm_test_file, num_agents, tester.env_settings)
    if tester.experiment == 'buttons':
        testing_env = MultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
    if tester.experiment == 'crafting':
        testing_env = MultiAgentCraftingEnv(tester.rm_test_file, num_agents, tester.env_settings)

    for i in range(num_agents):
        agent_list[i].reset_state()
        agent_list[i].initialize_reward_machine()

    s_team = np.full(num_agents, -1, dtype=int)
    a_team = np.full(num_agents, -1, dtype=int)
    u_team = np.full(num_agents, -1, dtype=int)

    for i in range(num_agents):
        s_team[i] = agent_list[i].s
    for i in range(num_agents):
        u_team[i] = agent_list[i].u

    testing_reward = 0

    trajectory = []

    step = 0
    accident_occured = False 

    # Starting interaction with the environment

    for t in range(testing_params.num_steps):
        step = step + 1

        # Perform a team step
        for i in range(num_agents):
            s, a = agent_list[i].get_next_action(-1.0, learning_params) # what is -1 in get next action?
            s_team[i] = s
            a_team[i] = a
            u_team[i] = agent_list[i].u

        # trajectory.append({'s' : np.array(s_team, dtype=int), 'a' : np.array(a_team, dtype=int), 'u_team': np.array(u_team, dtype=int), 'u': int(testing_env.u)})

        r, l, s_team_next = testing_env.environment_step(s_team, a_team)
        if l:
            trajectory.append(l)

        if not accident_occured:
            for j in l:
                if j in agent_list[1].local_event_set:
                    if j not in agent_list[1].individual_events:
                        accident_occured = True

        testing_reward = testing_reward + r


        ####  
        projected_l_dict = {}
        for i in range(num_agents):
            # Agent i's projected label is the intersection of l with its local event set

            projected_l_dict[i] = list(set(agent_list[i].local_event_set) & set(l))  # "&" is intersection in this case. local_event_set SHOULD NOT include events that are soley assinged to other agents, but it does include accidents. Agent sees "mistakes" that someone makes and their actions
            if len(projected_l_dict[i]) > 1: # this could happen when two people do accidents at the same time. 
                print("there are at least two evens: ", projected_l_dict[i], " It may be that all (or all but one) are accidents. ")
                projected_l_dict[i] = projected_l_dict[i][:1]
                
                acc_labels_observed_list = []
                for x in projected_l_dict:
                    if x not in agent_list[i].individual_events:
                        # Then x is an accident 
                        acc_labels_observed_list.append(x)


                if len(projected_l_dict[i]) - len(acc_labels_observed_list) > 1: 
                    raise ValueError(f"There were multiple( {len(projected_l_dict[i]) - len(acc_labels_observed_list)} ) non- accident events that occured.  agent {i}")
                else:
                    # there was only one non accident (and therefore at least one accident), lets just take the first one
                    projected_l_dict[i] = acc_labels_observed_list[:1]
                
            # Check if the event causes a transition from the agent's current RM state
            if not(agent_list[i].is_local_event_available(projected_l_dict[i])): # checks if my event is available to me. 
                projected_l_dict[i] = []
                # this could be the case if you have already messed up and "timber" or "arrived" happends because someone else did it. 


        for i in range(num_agents): #TODO: see trello, this is currently wrong. could throw out accidents in some situations

            # Enforce synchronization requirement on shared events. So I am not sure about this. In some ways, an event (like timber) via a string of accidents that would violate the expected synchronization. 
            # I also don't get it, since I don't think it was even possible for this to happen in Cyrus' experiments. 
            # CLAIM: synchronization will only fail if an accident has already happened (ex: timber because a3, so we are all lost already)
            # I could increase "local_event_set" to individual events which would only look at planned coordinations, but even then we don't

            #for event in projected_l_dict[i]:
            #    for j in range(num_agents):
            #        if (event in set(agent_list[j].local_event_set)) and (not (projected_l_dict[j] == projected_l_dict[i])): # the first condition would include an accident for multiple agents (as it should) but that they are not equal (which could happen with our arbitrarily taking the first accident. )
            #            projected_l_dict[i] = []


            # update the agent's internal representation
            # a = testing_env.get_last_action(i)

            agent_list[i].update_agent(s_team_next[i], a_team[i], r, projected_l_dict[i], learning_params, update_q_function=False)

        if testing_env.is_complete(): # Only breaking if agents successfully craft. 
            break

        #if all(agent.is_task_complete for agent in agent_list): # OLD. 
        #    break
    
    if show_print:
        print('Reward of {} achieved in {} steps. Current step: {} of {}'.format(testing_reward, step, tester.current_step, tester.total_steps))
        if (testing_reward > 0) and accident_occured: 
            print("     The team won by accident.    ")
        if (testing_reward > 0):
            print(f"     The team got  a reward of {testing_reward} in {step}  steps. Agent {testing_env.carrying_agent + 1} was transporting the log. ")

    return testing_reward, trajectory, step

def run_multi_agent_experiment(tester,
                            num_agents,
                            num_times,
                            show_print=True):
    """
    Run the entire q-learning with reward machines experiment a number of times specified by num_times.

    Inputs
    ------
    tester : Tester object
        Test object holding true reward machine and all information relating
        to the particular tasks, world, learning parameters, and experimental results.
    num_agents : int
        Number of agents in this experiment.
    num_times : int
        Number of times to run the entire experiment (restarting training from scratch).
    show_print : bool
        Flag indicating whether or not to output text to the terminal.
    """

    learning_params = tester.learning_params

    for t in range(num_times):
        # Reseting default step values
        tester.restart()

        rm_test_file = tester.rm_test_file
        rm_learning_file_list = tester.rm_learning_file_list

        # Verify that the number of local reward machines matches the number of agents in the experiment.
        assertion_string = "Number of specified local reward machines must match specified number of agents."
        assert (len(tester.rm_learning_file_list) == num_agents), assertion_string

        if tester.experiment == 'rendezvous':
            testing_env = MultiAgentGridWorldEnv(tester.rm_test_file, num_agents, tester.env_settings)
            num_states = testing_env.num_states

        if tester.experiment == 'buttons':
            testing_env = MultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
            num_states = testing_env.num_states

        if tester.experiment == 'crafting':
            testing_env = MultiAgentCraftingEnv(tester.rm_test_file, num_agents, tester.env_settings) # In decentralized learning, need to include extra things
            num_states = testing_env.num_states


        # Create the a list of agents for this experiment
        agent_list = []  #literally holds agents objects 
        for i in range(num_agents):
            actions = testing_env.get_actions(i)
            s_i = testing_env.get_initial_state(i)
            agent_list.append(Agent(rm_learning_file_list[i], s_i, num_states, actions, i, my_shared_events = tester.shared_events_dict[i], individual_events = tester.agent_event_spaces_dict[i] )) # It would probably be good to indlude extra information in the agent ? 

        num_episodes = 0
        step = 0

        # Task loop
        epsilon = learning_params.initial_epsilon

        while not tester.stop_learning():
            num_episodes += 1
            # epsilon = epsilon*0.99

            run_qlearning_task(epsilon,
                                tester,
                                agent_list,
                                show_print=show_print)

        # Backing up the results
        print('Finished iteration ',t)

    tester.agent_list = agent_list

    plot_multi_agent_results(tester, num_agents)

def plot_multi_agent_results(tester, num_agents):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25 = list()
    prc_50 = list()
    prc_75 = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps = list()

    plot_dict = tester.results['testing_steps']

    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))

        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps.append(step)

    plt.plot(steps, prc_25, alpha=0)
    plt.plot(steps, prc_50, color='red')
    plt.plot(steps, prc_75, alpha=0)
    plt.grid()
    plt.fill_between(steps, prc_50, prc_25, color='red', alpha=0.25)
    plt.fill_between(steps, prc_50, prc_75, color='red', alpha=0.25)
    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.locator_params(axis='x', nbins=5)

    plt.show()

