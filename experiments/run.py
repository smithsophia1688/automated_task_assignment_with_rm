
import pickle
from datetime import datetime
import os

if __name__ == "__main__":

    num_times = 1 # Number of separate trials to run the algorithm for

    num_agents = 3 # This will be automatically set to 3 for buttons experiment (max 10)

    # experiment = 'rendezvous'
    # experiment = 'centralized_rendezvous'
    # experiment = 'ihrl_rendezvous'
    # experiment = 'iql_rendezvous'

    experiment = 'buttons'
    # experiment = 'ihrl_buttons'
    # experiment = 'iql_buttons'

    if experiment == 'rendezvous':
        from config.rendezvous_config import rendezvous_config
        from dqprm import run_multi_agent_experiment
        tester = rendezvous_config(num_times, num_agents) # Get test object from config script
        run_multi_agent_experiment(tester, num_agents, num_times)

    if experiment == 'centralized_rendezvous':
        from config.rendezvous_config import rendezvous_config
        from run_centralized_coordination_experiment import run_centralized_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_centralized_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'ihrl_rendezvous':
        from config.rendezvous_config import rendezvous_config
        from run_ihrl_experiment import run_ihrl_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_ihrl_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'iql_rendezvous':
        from config.rendezvous_config import rendezvous_config
        from iql import run_iql_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_iql_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'buttons':
        from config.buttons_config import buttons_config
        from dqprm import run_multi_agent_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents) # Get test object from config script
        run_multi_agent_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'ihrl_buttons':
        from config.buttons_config import buttons_config
        from run_ihrl_experiment import run_ihrl_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents)
        run_ihrl_experiment(tester, num_agents, num_times, show_print=True)
    
    if experiment == 'iql_buttons':
        from config.buttons_config import buttons_config
        from iql import run_iql_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents)
        run_iql_experiment(tester, num_agents, num_times, show_print=True)


    # Save the results
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    data_path = os.path.join(parentDir, 'data')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    experiment_data_path = os.path.join(data_path, experiment)

    if not os.path.isdir(experiment_data_path):
        os.mkdir(experiment_data_path)

    now = datetime.now()
    save_file_str = now.strftime("%Y-%m-%d_%H-%M-%S_") + experiment + '.pkl'
    save_file = open(os.path.join(experiment_data_path, save_file_str), "wb")
    pickle.dump(tester, save_file)