from action import ALL_ACTIONS
from reward_system import compute_reward
def choose_greedy_action(current_grid):
    best_action = None
    best_value = -999999

    for action in ALL_ACTIONS:
        simulated = simulate_action_on_grid(current_grid, action)
        value = compute_reward(current_grid, simulated, 0)

        if value > best_value:
            best_value = value
            best_action = action

    return best_action


#WHAT MUST BE DONE FOR SIMULATION< THE BIGGEST STEP
#develop a system to call unity with a select move, unity simulates move in a dry run
#sends back (next state, reward, terminate, truncated), parse the inoformation
#recreate assignment AI's 