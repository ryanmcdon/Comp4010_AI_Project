import random
#formats the string for the action space in the format stanny used in discord. 
def format_action_string(action_dict):
    parts = []
    for key, value in action_dict.items():
        if value is True:
            parts.append(f"{key}")
        elif value is False:
            continue
        else:
            if isinstance(value, (int, float)) and value >= 0:
                parts.append(f"{key}: +{value}")
            else:
                parts.append(f"{key}: {value}")
    return "{ " + ", ".join(parts) + " }"


#this just generate random actions for testting as we dont have a ai yet
#i tried my best to remeber the possible actions and their ranges from our work period 
def random_action():

    possible_moves = [-4,-3,-2,-1, 0, +1, +2, +3, +4]


    possible_rotations = [-2,-1, 0, +1, +2]

    harddrop_choice = random.choice([True, False])

    action = {
        "move": random.choice(possible_moves),
        "rotate": random.choice(possible_rotations),
        "harddrop": harddrop_choice
    }
    print(format_action_string(action))
    return format_action_string(action)


