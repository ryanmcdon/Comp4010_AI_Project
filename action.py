ALL_ACTIONS = []
moves = [-4, -3, -2, -1, 0, +1, +2, +3, +4]
rots  = [-2, -1, 0, +1, +2]
drops = [True, False]

for m in moves:
    for r in rots:
        for d in drops:
            ALL_ACTIONS.append({
                "move": m,
                "rotate": r,
                "harddrop": d
            })
