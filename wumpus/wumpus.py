from argparse import ArgumentParser
from itertools import combinations
from json import dumps


# I/O
def log_arg(arg_name, value):
    print(f"[LOG] {arg_name}: {value}")

def add_clause(json_dict, json_key, json_value):
    if json_key in json_dict:
        raise ValueError(f"Key is already in use: {json_key}")
    json_dict[json_key] = json_value
    

# KEYS
COMMON_GROUNDG_KEY = "common_ground"
INITIAL_STATES_KEY = "initial_states"
#can be true 
INITIAL_STATES_CAN_BE_TRUE_KEY = "fluents"
INITIAL_STATES_CONDITIONS_KEY = "constraint"

ACTIONS_KEY = "actions"

#effects
EFFECTS_KEY = "effects"

#terminal
TERMINAL_STATE_FORMULA_KEY = "terminal"

#reward
REWARDS_KEY = "rewards"

# CONSTANTS
ALL_DIRECTIONS = [(0,1), (1,0), (0,-1), (-1,0)]


# Propostional Atoms and action names
def agent_positional_atom(row, column): # variable true when agent is at position (row, column) / proposition ?
    return f"agent_at_{row}_{column}"

def wumpus_positional_atom(row, column): # variable true when there is a wumpus at position (row, column) / proposition ?
    return f"wumpus_at_{row}_{column}"

def goal_positional_atom(row, column): # variable true when there is a goal at position (row, column) / proposition ?
    return f"goal_at_{row}_{column}"

def smell_positional_atom(row, column): # variable true when the position is smells (row, column) / proposition ?
    return f"smells_at_{row}_{column}"

def move_action_name(row_direction, column_direction): # name of move action
    if (row_direction, column_direction) == (0,1):
        dir = "east"
    elif (row_direction, column_direction) == (0,-1):
        dir = "west"
    elif (row_direction, column_direction) == (1,0):
        dir = "north"
    elif (row_direction, column_direction) == (-1,0):
        dir = "south"
    else:
        raise ValueError(f"Only the four cardinal directions have names: {row_direction=}, {column_direction=}")
    return f"move_{dir}"

def smell_action_name(): # name of smell action
    return "smell"
    
    
# Formulas
def top():
    return "T"

def bottom():
    return "⊥"

def positive_litteral(litteral):
    return litteral

def negative_litteral(litteral):
    return f"!{litteral}"

def logical_not(formula):
    return f"!({formula})"

def imply(premise, conclusion):
    return f"({premise})>({conclusion})"

def biimply(conclusion1, conclusion2):
    return f"({conclusion1})=({conclusion2})"

def term(litterals):
    return "&".join(f"{litteral}" for litteral  in litterals)

def clause(litterals):
    return "|".join(f"{litteral}" for litteral in litterals)

def logical_and(formulas):
    return "&".join(f"({formula})" for formula in formulas)

def logical_or(formulas):
    return "|".join(f"({formula})" for formula in formulas)

def explicit_belief(litteral):
    return f"△{litteral}"

def explicit_belief_formula(formula):
    return f"△({formula})"

def implicit_belief(literal):
    return f"□{literal}"

def implicit_belief_formula(formula):
    return f"□({formula})"



# Specification
def get_all_positions(nb_rows, nb_cols):
    return {(r, c) for r in range(nb_rows) for c in range(nb_cols)}

def valid_position_around(row, column, all_positions):
    return [(row + row_direction, column + column_direction) 
    for row_direction, column_direction in ALL_DIRECTIONS 
    if (row + row_direction, column + column_direction) in all_positions]

def not_all_atoms_false(atoms):
    return clause([positive_litteral(atom) for atom in atoms])

def not_all_atoms_true(atoms):
    return clause([negative_litteral(atom) for atom in atoms])

def at_least(k, atoms):
    n = len(atoms)
    # clauses: not (n - k + 1) false at the same time
    return [not_all_atoms_false(atms) for atms in combinations(atoms, n - k + 1)]

def at_most(k, atoms):
    n = len(atoms)
    # clauses: not (k + 1) true at the same time
    return [not_all_atoms_true(atms) for atms in combinations(atoms, k + 1)]

def generate_common_ground(nb_rows, nb_cols, nb_wumpus, random_start, random_goal):
    """ Generate the common ground that the agent knows and allowed to use to infer implicit knowledge
      from it's explicit knowledge.
      1. The agent knows the number of wumpus in the game.
      2. If there is a wumpus at a position then the valid position nears it smells.
      3. If it smells then there is at least one wumpus near.
    """
    
    formulas = []

    all_positions = get_all_positions(nb_rows, nb_cols)

    # - There are exactly nb_wumpus wumpus in the game.
    all_wumpus_propositional_atom = [wumpus_positional_atom(row, column) for row, column in all_positions]
   
    formulas += at_least(nb_wumpus, all_wumpus_propositional_atom)
    formulas += at_most(nb_wumpus, all_wumpus_propositional_atom)
    
    # - If there is a wumpus at a position then the valid position nears it smells.
    formulas += [imply(
                        wumpus_positional_atom(row,column),
                        term(smell_positional_atom(valid_row, valid_column)
                            for valid_row, valid_column in valid_position_around(row,column,all_positions)))
                        for row, column in all_positions]

    # - If it smells then there is at least one wumpus near
    formulas += [imply(
                    smell_positional_atom(row,column),
                    logical_or(wumpus_positional_atom(row, column) 
                        for row, column in valid_position_around(row,column,all_positions)))
                    for row, column in all_positions]

    return logical_and(formulas)

def generate_initial_states(nb_rows, nb_cols, nb_wumpus, common_ground=[], pseudo_safe=False):
    """ Generate the initial states of the world."""
   
    all_positions = get_all_positions(nb_rows, nb_cols)
    can_be_true = []
    conditions = []

    # --------------------------------------------------- ATOMS SET DEF ------------------------------------------------------------
    # define the propositionals atoms that encode the atoms possible set representing a valid initial state of the world
    
    initial_agent_possible_positions = all_positions
    all_agent_propositional_atom = [agent_positional_atom(row, column) for row, column in initial_agent_possible_positions]
    can_be_true += all_agent_propositional_atom

    initial_goal_possible_positions = all_positions
    all_goal_propositional_atom  = [goal_positional_atom(row, column) for row, column in initial_goal_possible_positions]
    can_be_true += [goal_positional_atom(row, column) for row, column in initial_goal_possible_positions]

    #Initial possible positions for wumpus
    wumpus_possible_positions = [(row, column) for row, column in all_positions]
    all_wumpus_propositional_atom = [wumpus_positional_atom(row, column) for row, column in wumpus_possible_positions]
    can_be_true += all_wumpus_propositional_atom

    #Initial possible smelly position near wumpuses
    smelly_atoms = set()
    for rw, rc in wumpus_possible_positions:
        possible_smelly_positions = valid_position_around(rw, rc, all_positions)
        for sr, sc in possible_smelly_positions:
            smelly_atoms.add(smell_positional_atom(sr,sc))
    can_be_true += sorted(smelly_atoms)

    # explicit knowledge about wumpus
    wumpus_info = [explicit_belief(wumpus_positional_atom(row, column)) for row, column in wumpus_possible_positions]
    can_be_true += wumpus_info
    
    # --------------------------------------------------- ATOMS SET CONDITIONS ------------------------------------------------------------
    # Add conditions on which those atoms can be mutualy true.
    
    if not random_start :
        #Initial agent position is bottom-left
        start_position = (0,0)
        conditions += [positive_litteral(agent_positional_atom(start_position[0], start_position[1]))]
        conditions += [term([negative_litteral(agent_positional_atom(raw,col)) for raw,col in all_positions if (raw,col) != start_position])]
    else:
         #Only one Initial agent position can be true
        conditions += at_least(1, all_agent_propositional_atom)
        conditions += at_most(1, all_agent_propositional_atom)

    if not random_goal :
        #Initial goal position is top-right
        goal_position = (nb_rows-1, nb_cols-1)
        conditions += [positive_litteral(goal_positional_atom(goal_position[0], goal_position[1]))]
        conditions += [term([negative_litteral(goal_positional_atom(raw,col)) for raw,col in all_positions if (raw,col) != goal_position])]
    else:
        # There only one goal position
        conditions += at_least(1, all_goal_propositional_atom)
        conditions += at_most(1, all_goal_propositional_atom)
    
    #There cannot be a wumpus at the agent starting position
    conditions += [
        clause([
        negative_litteral(wumpus_positional_atom(row,col)), 
        negative_litteral(agent_positional_atom(row,col))
        ])
        for row, col in initial_agent_possible_positions]
    
    #There cannot be a wumpus at the goal starting position
    conditions += [
        clause([
        negative_litteral(wumpus_positional_atom(row,col)), 
        negative_litteral(goal_positional_atom(row,col))
        ])
        for row, col in initial_goal_possible_positions]
    
    # If pseudo-safe: There cannot be a wumpus next to the goal or starting position
    if pseudo_safe:
        for row, col in initial_agent_possible_positions:
            for dxy in [-1,1]:
                if (row+dxy,col) in wumpus_possible_positions:
                    conditions.append(
                        clause([
                        negative_litteral(wumpus_positional_atom(row+dxy,col)), 
                        negative_litteral(agent_positional_atom(row,col))]))
                if (row,col+dxy) in wumpus_possible_positions:
                    conditions.append(
                        clause([
                        negative_litteral(wumpus_positional_atom(row,col+dxy)), 
                        negative_litteral(agent_positional_atom(row,col))]))
        for row, col in initial_goal_possible_positions:
            for dxy in [-1,1]:
                if (row+dxy,col) in wumpus_possible_positions:
                    conditions.append(
                        clause([
                        negative_litteral(wumpus_positional_atom(row+dxy,col)), 
                        negative_litteral(goal_positional_atom(row,col))]))
                if (row,col+dxy) in wumpus_possible_positions:
                    conditions.append(
                        clause([
                        negative_litteral(wumpus_positional_atom(row,col+dxy)), 
                        negative_litteral(goal_positional_atom(row,col))]))
    
    #The agent cannot start at the goal position
    conditions += [
        clause([
        negative_litteral(goal_positional_atom(row,col)), 
        negative_litteral(agent_positional_atom(row,col))
        ])
        for row, col in initial_agent_possible_positions
        if (row,col) in initial_agent_possible_positions 
        and (row,col) in initial_goal_possible_positions]
    
    #smells => wumpus near
    conditions += [
        imply(
            positive_litteral(smell_positional_atom(row,col)),
            clause([positive_litteral(wumpus_positional_atom(wumpus_row, wumpus_col)) 
                    for wumpus_row, wumpus_col in valid_position_around(row,col,all_positions)])
        )
    for row,col in wumpus_possible_positions]

    # ------------------------------------------- BELIEF SET DEF-------------------------------------------------

    # !△wumpus
    conditions += [negative_litteral(w) for w in wumpus_info]

    # △position
    can_be_true += [explicit_belief(
                    positive_litteral(agent_positional_atom(row, column))) 
                    for row, column in initial_agent_possible_positions]
    
    # △position <=> position
    conditions += [biimply(
                    explicit_belief(positive_litteral(agent_positional_atom(row, column))), 
                    positive_litteral(agent_positional_atom(row, column))) 
                        for row, column in initial_agent_possible_positions]

    # △smell
    can_be_true += [explicit_belief(
                    positive_litteral(smell_positional_atom(row, column))) 
                    for row, column in initial_agent_possible_positions]
    
    # △smell <=> position & wumpus near
    conditions += [
        biimply(
            # Then I know the position smells
            explicit_belief(positive_litteral(smell_positional_atom(row,column))),
            logical_and([
                # If I'm at a position
                positive_litteral(agent_positional_atom(row,column)),
                # and there is a wumpus near me
                clause(wumpus_positional_atom(near_row, near_column)
                for near_row, near_column in valid_position_around(row, column, all_positions)
                if (near_row, near_column) in wumpus_possible_positions)
                
            ])
        )
    for row, column in initial_agent_possible_positions]

    # △!smell
    can_be_true += [explicit_belief(
                    negative_litteral(smell_positional_atom(row, column))) 
                    for row, column in initial_agent_possible_positions]

    # △!smell <=> !wumpus near & position
    conditions += [biimply(
        # Then the agent knows the position is not smelly
        explicit_belief(negative_litteral(smell_positional_atom(row,column))),
        logical_and([
            # If there is no wumpus near the position
            term(negative_litteral(wumpus_positional_atom(near_row, near_column))
            for near_row, near_column in valid_position_around(row, column, all_positions)
            if (near_row, near_column) in wumpus_possible_positions),
            # and the agent is at the position
            positive_litteral(agent_positional_atom(row,column))
        ]))
    for row, column in initial_agent_possible_positions]

    spec = dict()
    add_clause(spec, INITIAL_STATES_CAN_BE_TRUE_KEY, can_be_true)
    add_clause(spec, INITIAL_STATES_CONDITIONS_KEY, logical_and(conditions + common_ground))
    
    return spec
               
def move_action(row_direction, column_direction, all_positions,nb_rows,nb_cols):
    #ideas:
    # action to infer wumpus !wumpus
    # action to forget smell & !smell

    # North means row += 1 (rows numbered bottom-up)
    spec = dict()

    # Implementation: add new position and del old one if different (agent stays in place when hitting a wall)
    direction = (row_direction,column_direction)
    for pos in all_positions:
        match direction:
            case(0,1):#east
                if (pos[1] == 0 and pos[1] == nb_cols-1) is False: 
                    if pos[1] == 0: #BorderLeft 
                        spec.update(border_leaving_movement_atom_effect(pos))
                        spec.update(border_leaving_movement_belief_effect(pos))
                    elif pos[1] == nb_cols-1: #BorderRight
                        spec.update(border_reaching_or_staying_movement_atom_effect(pos,direction))
                        spec.update(border_reaching_or_staying_movement_belief_effect(pos,direction))
                    else:
                        spec.update(movement_atom_effect(pos,direction))
                        spec.update(movement_belief_effect(pos,direction))

            case(0,-1):#west
                if (pos[1] == 0 and pos[1] == nb_cols-1) is False:
                    if pos[1] == nb_cols-1: #BorderRight
                        spec.update(border_leaving_movement_atom_effect(pos))
                        spec.update(border_leaving_movement_belief_effect(pos))
                    elif pos[1] == 0: #BorderLeft
                        spec.update(border_reaching_or_staying_movement_atom_effect(pos,direction))
                        spec.update(border_reaching_or_staying_movement_belief_effect(pos,direction))
                    else:
                        spec.update(movement_atom_effect(pos,direction))
                        spec.update(movement_belief_effect(pos,direction))

            case(1,0):#north
                if (pos[0] == 0 and pos[0] == nb_rows-1) is False:
                    if pos[0] == 0: #BorderBottom
                        spec.update(border_leaving_movement_atom_effect(pos))
                        spec.update(border_leaving_movement_belief_effect(pos))
                    elif pos[0] == nb_rows-1: #BorderTop
                        spec.update(border_reaching_or_staying_movement_atom_effect(pos,direction))
                        spec.update(border_reaching_or_staying_movement_belief_effect(pos,direction))
                    else:
                        spec.update(movement_atom_effect(pos,direction))
                        spec.update(movement_belief_effect(pos,direction))

            case(-1,0):#south
                if (pos[0] == 0 and pos[0] == nb_rows-1) is False:
                    if pos[0] == nb_rows-1: #BorderTop
                        spec.update(border_leaving_movement_atom_effect(pos))
                        spec.update(border_leaving_movement_belief_effect(pos))
                    elif pos[0] == 0: #BorderDown
                        spec.update(border_reaching_or_staying_movement_atom_effect(pos,direction))
                        spec.update(border_reaching_or_staying_movement_belief_effect(pos,direction))
                    else:
                        spec.update(movement_atom_effect(pos,direction))
                        spec.update(movement_belief_effect(pos,direction))
            case _:
                raise ValueError(f"Only the four cardinal directions are considered as movement actions: {row_direction=}, {column_direction=}")

    # - Add belief new smell
    spec.update({explicit_belief(smell_positional_atom(row+row_direction, column+column_direction))
                :
                logical_or([
                    # I already knew it
                    explicit_belief(smell_positional_atom(row+row_direction, column+column_direction)),
                    # I discovered it
                    logical_and([
                        positive_litteral(agent_positional_atom(row, column)),
                        logical_or([wumpus_positional_atom(neighbor_row, neighbor_column) 
                                for neighbor_row, neighbor_column in valid_position_around(row+row_direction,column+column_direction, all_positions)])])])
                for row, column in all_positions 
                if (row + row_direction, column + column_direction) in all_positions })
    
    # - Add belief new not smell
    spec.update({explicit_belief(negative_litteral(smell_positional_atom(row+row_direction, column+column_direction)))      
                : 
                logical_or([
                    #I already knew it
                    explicit_belief(negative_litteral(smell_positional_atom(row+row_direction, column+column_direction))),
                    logical_and([
                        positive_litteral(agent_positional_atom(row, column)),
                        logical_and([negative_litteral(wumpus_positional_atom(neighbor_row, neighbor_column))
                                for neighbor_row, neighbor_column in valid_position_around(row+row_direction, column+column_direction, all_positions)])])])  
                for row, column in all_positions 
                if (row + row_direction, column + column_direction) in all_positions })
    
    return spec

def border_leaving_movement_atom_effect(pos):
    return {agent_positional_atom(pos[0],pos[1]) : bottom()}

def border_leaving_movement_belief_effect(pos):
    return {explicit_belief(agent_positional_atom(pos[0],pos[1])) : bottom()}

def border_reaching_or_staying_movement_atom_effect(pos,direction):
    return {agent_positional_atom(pos[0],pos[1])
                :logical_or([
                    agent_positional_atom(pos[0]-direction[0],pos[1]-direction[1]),
                    agent_positional_atom(pos[0],pos[1])
                ])
            }

def border_reaching_or_staying_movement_belief_effect(pos,direction):
    return {explicit_belief(agent_positional_atom(pos[0],pos[1]))
                    :logical_or([
                        agent_positional_atom(pos[0]-direction[0],pos[1]-direction[1]),
                        agent_positional_atom(pos[0],pos[1])
                    ])
                }

def movement_atom_effect(pos,direction):
    return {agent_positional_atom(pos[0],pos[1])
                : agent_positional_atom(pos[0]-direction[0],pos[1]-direction[1])}

def movement_belief_effect(pos,direction):
    return {explicit_belief(agent_positional_atom(pos[0],pos[1]))
                : agent_positional_atom(pos[0]-direction[0],pos[1]-direction[1])}

def generate_actions(nb_rows, nb_cols):
    """ generate the actions spec of the game."""
    
    all_positions = get_all_positions(nb_rows, nb_cols)
    
    spec = dict()
    #Movement actions
    for row_direction, column_direction in ALL_DIRECTIONS:
        action_name = move_action_name(row_direction, column_direction)
        noeff_positions = [agent_positional_atom(r,c) for (r,c) in all_positions if (r+row_direction, c+column_direction) not in all_positions]
        safe_shield = []
        for (agent_row, agent_col) in all_positions:
            successor_row = agent_row + row_direction
            successor_col = agent_col + column_direction
            if (successor_row, successor_col) in all_positions:
                shield = logical_or([negative_litteral(explicit_belief(agent_positional_atom(agent_row, agent_col))), implicit_belief(negative_litteral(wumpus_positional_atom(successor_row,successor_col)))])
                safe_shield.append(shield)


        spec[action_name] = {
            "effects": move_action(row_direction, column_direction, all_positions,nb_rows,nb_cols),
            "shields": {
                "can_move": logical_and(logical_not(explicit_belief(p)) for p in noeff_positions),
                "is_safe": logical_and(safe_shield)
            }
        }
    # forgetting and inferring wumpus possitions action
    spec.update(forget_and_infer_action(nb_rows,nb_cols))
    # noop action
    spec["noop"]={"effects":{}}

    return spec


def forget_and_infer_action(nb_rows, nb_cols):
    effects = {}
    for row,col in get_all_positions(nb_rows, nb_cols):
        foo = tuple(negative_litteral(explicit_belief(wumpus_positional_atom(row+dr,col+dc))) for dr in [-1,0,1] for dc in [-1,0,1] if 0 <= row+dr < nb_rows and 0 <= col+dc < nb_cols and dr*dc== 0 and dr!=dc)
        effects.update({explicit_belief(smell_positional_atom(row,col)): logical_and((explicit_belief(smell_positional_atom(row,col)), logical_or((explicit_belief(agent_positional_atom(row,col)),
        logical_and(foo)))))})
        effects.update({explicit_belief(logical_not(smell_positional_atom(row,col))): logical_and((explicit_belief(logical_not(smell_positional_atom(row,col))), explicit_belief(agent_positional_atom(row,col))))})
    for row,col in get_all_positions(nb_rows, nb_cols):
        effects.update({explicit_belief(wumpus_positional_atom(row,col)):
            logical_or([explicit_belief(wumpus_positional_atom(row,col)), implicit_belief(wumpus_positional_atom(row,col))])})
    return {"forget_and_infer": {"effects": effects}}


def goal_reached(row,column):
    return term(
            [positive_litteral(agent_positional_atom(row, column)),
            positive_litteral(goal_positional_atom(row, column))])

def  generate_terminal_state_formula(nb_rows, nb_cols, random_goal):
    """ generate the terminal state formula of the game."""
    spec = []
    #Games ends if:
    # - the agent is at the goal
    if  not   random_goal: spec += [logical_or([goal_reached(row, column) for row, column in [(nb_rows-1,nb_cols-1)]])]
    else:               spec += [logical_or([goal_reached(row, column) for row, column in get_all_positions(nb_rows, nb_cols)])]

    # - the agent is dead (agent at the same place as a wumpus)
    spec += [logical_or([
                term([positive_litteral(agent_positional_atom(row, column)), 
                      positive_litteral(wumpus_positional_atom(row, column))]) 
                      for row, column in get_all_positions(nb_rows, nb_cols)
                      ])]
    
    return logical_or(spec)

def generate_rewards(nb_rows, nb_cols):
    """ generate the rewards of the game as a dict of condition formula and reward value."""
    # reward 1 for arriving at the goal
    goal_rewards = []
    if not random_goal:
        goal_rewards.append({"post": goal_reached(nb_rows-1, nb_rows-1), "reward": 1})
    else:
        for row, column in get_all_positions(nb_rows, nb_cols):
            goal_rewards.append({"post": goal_reached(row, column), "reward": 1})
    # reward -1 for running into the wumpus
    wumpus_rewards = []
    for row, column in get_all_positions(nb_rows, nb_cols):
        wumpus_rewards.append({"post": term([positive_litteral(agent_positional_atom(row, column)), 
                       positive_litteral(wumpus_positional_atom(row, column))]), "reward" : -1})
    # reward for finding the wumpus
    find_wumpus_rewards = []
    for (row,col) in get_all_positions(nb_rows,nb_cols):
        find_wumpus_rewards.append({"pre": negative_litteral(explicit_belief(wumpus_positional_atom(row,col))),
                     "post": explicit_belief(wumpus_positional_atom(row,col)), "reward": 0.1 })
    new_pos_smell_rewards = []
    for (row,col) in get_all_positions(nb_rows,nb_cols):
        new_pos_smell_rewards.append({"pre": logical_and((negative_litteral(explicit_belief(smell_positional_atom(row,col))),)+
            tuple(negative_litteral(explicit_belief(wumpus_positional_atom(row+dr,col+dc))) for dr in [-1,0,1] for dc in [-1,0,1] if 0 <= row+dr < nb_rows and 0 <= col+dc < nb_cols and dr*dc== 0 and dr!=dc)),
                     "post": explicit_belief(smell_positional_atom(row,col)), "reward": 0.01 })
    new_neg_smell_rewards = []
    for (row,col) in get_all_positions(nb_rows,nb_cols):
        new_neg_smell_rewards.append({"pre": negative_litteral(explicit_belief(negative_litteral(smell_positional_atom(row,col)))),
                     "post": explicit_belief(negative_litteral(smell_positional_atom(row,col))), "reward": 0.01 })

    return {"arrive_at_goal": goal_rewards, "run_into_wumpus": wumpus_rewards, "new_pos_smell": new_pos_smell_rewards, "new_neg_smell": new_neg_smell_rewards, "infer_wumpus_position": find_wumpus_rewards}


if __name__ == "__main__":
    #add param forgetting agent

    # Command-line arguments
    parser = ArgumentParser()
    parser.add_argument("nb_rows", help="Number of rows", type=int)
    parser.add_argument("nb_cols", help="Number of columns", type=int)
    parser.add_argument("nb_wumpus", help="Number of wumpuses", type=int)
    parser.add_argument("--pseudo_safe", help="cells next to start and next to goal are always free", action="store_true")
    parser.add_argument("--random_start", help="starting position fixed at bottom left", action="store_true")
    parser.add_argument("--random_goal", help="goal position fixed at top right",  action="store_true")

    args = parser.parse_args()
    nb_rows = args.nb_rows
    nb_cols = args.nb_cols
    nb_wumpus = args.nb_wumpus
    random_start = args.random_start
    random_goal = args.random_goal
    pseudo_safe = args.pseudo_safe
    #log_arg("Number of rows", nb_rows)
    #log_arg("Number of cols", nb_cols)
    #log_arg("Number of wumpus", nb_wumpus)
    #log_arg("random start", random_start)
    #log_arg("random_goal", random_goal)
    #log_arg("forgetting_agent", forgetting_agent)

    # Writing spec.
    spec = dict()
    
    common_ground = generate_common_ground(nb_rows, nb_cols, nb_wumpus, random_start, random_goal)
    initial_states = generate_initial_states(nb_rows, nb_cols, nb_wumpus, [common_ground], pseudo_safe)
    actions_spec = generate_actions(nb_rows, nb_cols)
    terminal_state_formula = generate_terminal_state_formula(nb_rows, nb_cols, random_goal)
    rewards_spec = generate_rewards(nb_rows, nb_cols)
    
    add_clause(spec, INITIAL_STATES_KEY, initial_states)
    add_clause(spec, COMMON_GROUNDG_KEY, common_ground)
    add_clause(spec, ACTIONS_KEY, actions_spec)
    add_clause(spec, TERMINAL_STATE_FORMULA_KEY, terminal_state_formula)
    add_clause(spec, REWARDS_KEY, rewards_spec)
    
    json = dumps(spec, ensure_ascii=False, indent=4)
    print(json)
