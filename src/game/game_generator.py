import yaml
import numpy as np
from .spec import *
from .game import Game


def read_game(file_path):
    input = yaml.safe_load(open(file_path))
    errors = []
    err = errors.append

    # --- Hosts ---
    hosts = {h["name"]: Host(h["name"], h["impact"]) for h in input["hosts"]}
    zones = {z["name"]: Zone(z["name"], z["eta"], z["hosts"]) for z in input["zones"]}

    # --- Attacker actions ---
    edges = [Edge(e["src"], e["dst"], e["vuln"]) for e in input["edges"]]
    vulns = {v["name"]: Vuln(v["name"], v["success"]) for v in input["vulns"]}

    # --- Defender actions ---
    controls = {}
    for c in input["controls"]:
        mitigations = {m["vuln"]: m["factor"] for m in c["mitigations"]}
        controls[c["name"]] = Control(c["name"], c["cost"], mitigations)

    # --- Model ---
    foothold = {f["host"]: f["weight"] for f in input["model"]["initial_foothold"]}
    gamma = input["model"]["discount"]

    if sum(z.detection for z in zones.values()) > 1.0 + 1e-9:
        err(
            f"zone etas sum to {sum(z.detection for z in zones.values()):.3f}, must be <= 1"
        )

    if sum(p for p in foothold.values()) > 1.0 + 1e-9:
        err(
            f"foothold initial belief sums to {sum(p for p in foothold.values()):.3f}, must be <= 1"
        )

    if errors:
        raise SpecError("invalid input:\n  " + "\n  ".join(errors))

    spec = Spec(
        hosts=hosts,
        zones=zones,
        edges=edges,
        vulns=vulns,
        controls=controls,
        foothold=foothold,
        gamma=gamma,
    )
    return Game(spec)


def write_hsvi(file_name, game):
    # names must contain no whitespace, otherwise treated as new param in posg file

    n_states, n_defences, n_attacks, n_observations, discount = (
        len(game.states),
        len(game.defences),
        len(game.attacks),
        len(game.observations),
        game.gamma,
    )
    body = []

    # Section 2. states: "name partition" -> always partition 0
    for i, s in enumerate(game.states.to_key):
        body.append(f"State{i}[{''.join(map(str, s))}] 0")

    # Section 3. defender actions (p1): "name"
    for i, cfg in enumerate(game.defences.to_key):
        body.append(f"[Configuration{i}[{','.join(cfg)}]]")

    # Section 4. attack actions (p2): "name"
    for i, (src, dst, v) in enumerate(game.attacks.to_key[: game.n_exploits]):
        body.append(f"[Edge{i}[{src}>{dst}:{v}]]")
    body.append(f"[End]")

    # Section 5. observations: "name"
    for i, obs in enumerate(game.observations.to_key):
        body.append(f"[Observation{i}[{obs}]]")

    # Section 6. actions per state (p2): list of index of actions, each line is for a state
    for s in range(n_states):
        body.append(" ".join(map(str, np.flatnonzero(game.attacker_legal[s]))))

    # Section 7. actions per partition (p1): only one partition, list of all actions' indexes
    body.append(" ".join([str(d) for d in range(n_defences)]))

    # Section 8. transitions: [s] [d] [a] [o] [s'] [prob]
    obs_support = [np.flatnonzero(game.O[s2]) for s2 in range(n_states)]
    n_trans = 0
    for s in range(n_states):
        for d in range(n_defences):
            for a in np.flatnonzero(game.attacker_legal[s]):
                for s2 in np.flatnonzero(game.T[s, d, a]):
                    for o in obs_support[s2]:
                        p = game.T[s, d, a, s2] * game.O[s2, o]
                        body.append(f"{s} {d} {a} {o} {s2} {p:.10g}")
                        n_trans += 1

    # Section 9. rewards (p1): [s] [d] [a] [reward] (expected immediate reward)
    n_rewards = 0
    for s in range(n_states):
        for d in range(n_defences):
            for a in np.flatnonzero(game.attacker_legal[s]):
                r = game.R[s, d, a]
                if abs(r) > 1e-12:
                    body.append(f"{s} {d} {a} {r:.10g}")
                    n_rewards += 1

    # Section 10. initial belief: [initial partition] [prob per state]
    body.append(" ".join(map(str, game.initial_belief)))

    # Section 1. game description: |S| |partition| |defences| |attacks| |observations| |transitions| |rewards| discount
    # done last, after computing number of transitions and rewards
    header = f"{n_states} 1 {n_defences} {n_attacks} {n_observations} {n_trans} {n_rewards} {discount:.4f}"
    with open(f"hsvi/{file_name}.posg", "w") as file:
        file.write(f"{header}\n{'\n'.join(body)}")
