import yaml
import numpy as np
from itertools import combinations
from collections import deque
from .spec import *


# Used for everything index needed in the game file for hsvi
class IndexRegistry:
    def __init__(self):
        self.to_idx = {}
        self.to_key = []

    def intern(self, key):
        if key not in self.to_idx:
            self.to_idx[key] = len(self.to_key)
            self.to_key.append(key)
        return self.to_idx[key]

    def __len__(self):
        return len(self.to_key)


class Game:
    spec: Spec

    # --- HSVI index spaces ---
    # boolean vector (tuple of 1 and 0) as keys
    states: IndexRegistry
    # edges (tuple of src, dst and vuln) as keys
    attacks: IndexRegistry
    # combinations of controls (tuple of names) as keys
    defences: IndexRegistry
    # zones' names or none as keys
    observations: IndexRegistry

    # attack actions 0..n_exploits-1 are real exploits
    n_exploits: int
    # attack_end == n_exploits, used when algorithm needs infinite horizon (HSVI, SAB)
    attack_end: int

    # --- Rest of Game Model ---
    gamma: float
    # (S, A) bool: attack action a is playable in state s
    attacker_legal: np.ndarray
    # (s, d, a, s2) -> P(s2 | s, d, a), shape (S, D, A, S)
    T: np.ndarray
    # (s, o) -> P(o | s), shape (S, Z); the observation depends on the landing state only
    O: np.ndarray
    # (s, d, a) -> expected immediate reward, shape (S, D, A)
    R: np.ndarray
    # s -> belief, shape (S,)
    initial_belief: np.ndarray

    # --- Helpers ---
    # h -> host idx
    _h_idx: dict[str, int]
    # h -> z
    _zone_of: dict[str, str]
    # d -> a -> success probability of mitigated attack, as indexes
    _p_success: list[list[float]]
    # (s, a) -> s2 if the exploit succeeds, useful for transitions without recalculating them, as indexes
    _successors: dict[tuple[int, int], int]
    # s -> [(s2, success prob), ...] useful for likelihood walk
    _out: list[tuple[int, float], ...]
    # d -> combined cost of controls in d
    _costs: list[float]
    # s -> impact of landing in s
    _impacts: list[float]
    # base_s -> s -> likelihood for each base state, likelihood of every reachable states
    _likelihoods: list[dict[int, float]]
    # s -> risk of s
    _risks: list[float]

    def __init__(self, spec):
        self.spec = spec

        self.states = IndexRegistry()
        self.attacks = IndexRegistry()
        self.defences = IndexRegistry()
        self.observations = IndexRegistry()

        self.gamma = self.spec.gamma

        self._h_idx = {h.name: i for i, h in enumerate(spec.hosts.values())}
        self._zone_of = {h: z.name for z in self.spec.zones.values() for h in z.hosts}
        self._p_success = []
        self._successors = {}
        self._out = []
        self._costs = []
        self._impacts = []
        self._likelihoods = []
        self._risks = []

        self._build_game()
        self._validate()

    def _build_game(self):
        self._build_actions()
        self._explore_states()
        self._build_observations()
        self._build_p_success()
        self._build_transitions()
        self._build_rewards()
        self._build_inital_belief()

    def _build_actions(self):
        # one per exploit edge
        for e in self.spec.edges:
            self.attacks.intern(e)
        self.n_exploits = len(self.attacks)
        self.attack_idle = self.attacks.intern("idle")

        for i in range(len(self.spec.controls) + 1):
            for action in combinations(self.spec.controls.keys(), i + 1):
                self.defences.intern(action)
                self._costs.append(sum(self.spec.controls[d].cost for d in action))

    def _explore_states(self):
        # first add foothold hosts, then walk the graph to add other states
        for h in self.spec.foothold.keys():
            self.states.intern(tuple(1 if x == h else 0 for x in self.spec.hosts))

        legal_lists = []
        s = 0
        while s < len(self.states):  # registry grows while we walk
            state = self.states.to_key[s]
            legal, out = [], []
            # [:self.n_exploits] to not pass over end idle action
            for a, e in enumerate(self.attacks.to_key[: self.n_exploits]):
                if state[self._h_idx[e.src]] == 1 and state[self._h_idx[e.dst]] == 0:
                    new_state = list(state)
                    new_state[self._h_idx[e.dst]] = 1
                    s2 = self.states.intern(tuple(new_state))
                    self._successors[(s, a)] = s2
                    legal.append(a)
                    out.append((s2, self.spec.vulns[e.vuln].success))
            legal_lists.append(legal)
            self._out.append(out)
            s += 1

        # Add end idle action to last state
        for s in range(len(self.states)):
            if not legal_lists[s]:
                legal_lists[s] = [self.attack_idle]
                self._successors[(s, self.attack_idle)] = s

        # boolean mask instead of a ragged list of lists
        self.attacker_legal = np.zeros(
            (len(self.states), len(self.attacks)), dtype=bool
        )
        for s, acts in enumerate(legal_lists):
            self.attacker_legal[s, acts] = True

    def _get_obs_distribution(self, s):
        state = self.states.to_key[s]
        Zs = {self._zone_of[h] for h in self.spec.hosts if state[self._h_idx[h]] == 1}
        dist = [(self.observations.to_idx[z], self.spec.zones[z].detection) for z in Zs]
        dist.append(
            (
                self.observations.to_idx["none"],
                1 - sum(self.spec.zones[z].detection for z in Zs),
            )
        )
        return [(o, p) for o, p in dist if p > 0]

    def _build_observations(self):
        for z in self.spec.zones.keys():
            self.observations.intern(z)
        self.observations.intern("none")

        self.O = np.zeros((len(self.states), len(self.observations)))
        for s in range(len(self.states)):
            for o, p in self._get_obs_distribution(s):
                self.O[s, o] = p

    def _build_p_success(self):
        self._p_success = []
        for d in self.defences.to_key:
            p_successes = []
            for a in self.attacks.to_key[: self.n_exploits]:
                p_success = 1.0
                for c in d:
                    migitation = self.spec.controls[c].mitigations.get(a.vuln, 0.0)
                    p_success *= 1 - migitation
                p_successes.append(self.spec.vulns[a.vuln].success * p_success)
            p_successes.append(1.0)  # idle action
            self._p_success.append(p_successes)

    def _build_transitions(self):
        S, D, A = len(self.states), len(self.defences), len(self.attacks)
        self.T = np.zeros((S, D, A, S))
        for s in range(S):
            for d in range(D):
                for a in range(A):
                    if self.attacker_legal[s, a]:
                        target = self._successors[(s, a)]
                        p_success = self._p_success[d][a]
                        # += so the idle action (target == s, p_success == 1) stays valid, otherwise rewritten with 1 - 1 = 0
                        self.T[s, d, a, target] += p_success
                        self.T[s, d, a, s] += 1 - p_success
                    else:
                        # unplayable: harmless self-loop, masked by attacker_legal
                        self.T[s, d, a, s] = 1.0

    def _build_impact(self):
        for s in range(len(self.states)):
            state = self.states.to_key[s]
            impact = sum(
                h.impact
                for h in self.spec.hosts.values()
                if state[self._h_idx[h.name]] == 1
            )
            self._impacts.append(impact)

    def _get_likelihoods(self, base_s):
        likelihoods, prod = {}, {}
        queue, queued = deque([base_s]), {base_s}
        while queue:
            s = queue.popleft()
            likelihoods[s] = 1.0 if s == base_s else 1.0 - prod[s]
            for s2, p in self._out[s]:
                prod[s2] = prod.get(s2, 1.0) * (1.0 - likelihoods[s] * p)
                if s2 not in queued:
                    queued.add(s2)
                    queue.append(s2)
        return likelihoods

    def _build_likelihood(self):
        for s in range(len(self.states)):
            self._likelihoods.append(self._get_likelihoods(s))

    def _build_risk(self):
        self._build_impact()
        self._build_likelihood()

        for s in range(len(self.states)):
            likelihoods = self._likelihoods[s]
            self._risks.append(
                sum(likelihoods[s2] * self._impacts[s2] for s2 in likelihoods.keys())
            )

    def _build_rewards(self):
        self._build_risk()

        max_impact = max(self._impacts)
        max_risk = max(self._risks)
        max_cost = max(self._costs)

        S, D, A = len(self.states), len(self.defences), len(self.attacks)
        self.R = np.zeros((S, D, A))
        for s in range(S):
            for d in range(D):
                for a in np.flatnonzero(self.attacker_legal[s]):
                    cost = self._costs[d]
                    total = 0.0
                    for s2 in np.flatnonzero(self.T[s, d, a]):
                        p = self.T[s, d, a, s2]
                        reward = -(self._risks[s2] / max_risk) - (cost / max_cost)
                        if s != s2:
                            reward -= self._impacts[s2] / max_impact
                        total += p * reward
                    self.R[s, d, a] = total

    def _build_inital_belief(self):
        belief_list = []
        for s in self.states.to_key:
            belief = 0.0
            for h, p in self.spec.foothold.items():
                if s == tuple(1 if x == h else 0 for x in self.spec.hosts):
                    belief = p
            belief_list.append(belief)
        self.initial_belief = np.array(belief_list)

    def _validate(self, tol=1e-9):
        errors = []
        if not np.allclose(self.T.sum(axis=3), 1.0, atol=tol):
            errors.append("some T[s, d, a, :] does not sum to 1")
        if not np.allclose(self.O.sum(axis=1), 1.0, atol=tol):
            errors.append(
                "some O[s, :] does not sum to 1 (zone detection rates sum to > 1?)"
            )
        if abs(self.initial_belief.sum() - 1.0) > tol:
            errors.append("initial belief does not sum to 1")
        if not self.attacker_legal.any(axis=1).all():
            errors.append("a state has no legal attacker action")
        if errors:
            raise ValueError("invalid game:\n  " + "\n  ".join(errors))


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
