import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_multi_agent_results(ax,best_tester, label=" ",color="red", what_to_plot = 'testing_steps', window = 10):
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

    plot_dict_1 = best_tester.results[what_to_plot]
   
   
    for step in plot_dict_1.keys():
        if len(current_step) < window:
            current_25.append(np.percentile(np.array(plot_dict_1[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict_1[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict_1[step]), 75))
            current_step.append(sum(plot_dict_1[step])/len(plot_dict_1[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_1[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_1[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_1[step]),75))
            current_step.append(sum(plot_dict_1[step])/len(plot_dict_1[step]))

        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps.append(step)

    
    ax.plot(steps, prc_25, alpha=0)
    ax.plot(steps, prc_50, color=color, label=label)
    ax.plot(steps, prc_75, alpha=0)
    ax.grid()
    ax.fill_between(steps, prc_50, prc_25, color=color, alpha=0.25)
    ax.fill_between(steps, prc_50, prc_75, color=color, alpha=0.25)
    
    ax.locator_params(axis='x', nbins=5)
    
#best_file_name = '2022_11_29-11.12.15_AM_best.pkl'

best_file_name = '2022_11_29-11.29.48_AM_best.pkl' #previous 

# best_file_name = '2022_11_29-01.28.28_PM_best.pkl' #only 70000 steps
best_full_file_name = 'data/saved_testers/five_agent_task/' + best_file_name 

trivial_file_name = '2022_11_29-11.37.09_AM_trivial.pkl'
# trivial_file_name = '2022_11_29-01.31.10_PM_trivial.pkl' # only 70000 steps
trivial_full_file_name = 'data/saved_testers/five_agent_task/' + trivial_file_name 

with open(best_full_file_name , 'rb') as f :
    best_tester = pickle.load(f)

with open(trivial_full_file_name , 'rb') as f :
    trivial_tester = pickle.load(f)

print(type(best_tester))
print(type(trivial_tester))



# what_to_plot = 'discounted_reward'
# what_to_plot = 'testing_steps'
what_to_plot = 'testing_steps'
save = True

if what_to_plot == 'discounted_reward':
    fig, ax = plt.subplots()
    window = 1
    plot_multi_agent_results(ax, best_tester, label='ATAD', color= 'red', what_to_plot = what_to_plot , window = window)
    plot_multi_agent_results(ax, trivial_tester, label = 'Baseline', color = 'blue',  what_to_plot = what_to_plot, window = window)
    ax.set_ylabel('Discounted Reward', fontsize=15)
    ax.set_xlabel('Training Steps', fontsize=15)
    plt.title(" ")
    plt.legend()
    import tikzplotlib
    tikzplotlib.save("data/saved_plots/five_plot_discount_reward.tex")

elif what_to_plot == 'testing_steps':
    fig, ax = plt.subplots()
    window = 10
    plot_multi_agent_results(ax, best_tester, label= 'ATAD', color= 'red', what_to_plot = what_to_plot, window = window)
    plot_multi_agent_results(ax, trivial_tester, label = 'Baseline', color = 'blue',  what_to_plot = what_to_plot, window = window)
    ax.set_ylabel('Testing Steps To Task Completion', fontsize=15)
    ax.set_xlabel('Training Steps', fontsize=15)
    plt.legend()
    import tikzplotlib
    tikzplotlib.save("data/saved_plots/five_plot_testing_steps.tex")


elif what_to_plot == 'both':
    fig, (ax1, ax2) = plt.subplots(2)
    window_dr = 1
    window_ts = 10
    plot_multi_agent_results(ax1, best_tester, label='ATAD', color= 'red', what_to_plot = 'testing_steps' , window = window_ts)
    plot_multi_agent_results(ax1, trivial_tester, label = 'Baseline', color = 'blue',  what_to_plot = 'testing_steps', window = window_ts)
    ax1.set_ylabel(f'Testing Steps \n To Task Completion', fontsize=15)

    plot_multi_agent_results(ax2, best_tester, label='ATAD', color= 'red', what_to_plot = 'discounted_reward' , window = window_dr)
    plot_multi_agent_results(ax2, trivial_tester, label = 'Baseline', color = 'blue',  what_to_plot = 'discounted_reward', window = window_dr)
    ax2.set_ylabel('Discounted Reward', fontsize=15)
    ax2.set_xlabel('Training Steps', fontsize=15)

    plt.title(" ")
    plt.legend()
    # fig.suptitle('Vertically stacked subplots')
    # plt.show()
    # if save:
    import tikzplotlib
    tikzplotlib.save("data/saved_plots/five_plot_vertically_stacked.tex")


    
