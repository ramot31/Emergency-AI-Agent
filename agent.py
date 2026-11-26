"""Orchestrateur de l'agent (Bloc 1 + Bloc 6) : relie ML/DL, logique, recherche et planification."""
from facts import state_to_facts, fact_str
from interface import apply_event
import stubs as S


class Agent:
    def __init__(self, state, ml=S.ml_predict, dl=S.dl_predict, reasoner=S.reason,
                 searcher=S.search_allocation, planner=S.make_plan):
        self.state, self.ml, self.dl = state, ml, dl
        self.reasoner, self.searcher, self.planner = reasoner, searcher, planner
        self.plan, self.plan_version, self.step = [], None, 0
        self.completed = set()          # patients dont le plan est entièrement exécuté
        self.journal = []

    def log(self, etape, msg):
        line = f"[{etape}] {msg}"
        self.journal.append(line); print(line)

    # ------------------------------------------------------------------ un cycle complet
    def cycle(self):
        self.log("CYCLE", f"Nouveau cycle de décision sur l'état v{self.state.version}")
        self.log("ETAT", self.state.summary())
        waiting = self.state.waiting_patients()
        facts = state_to_facts(self.state)

        # 1) Evidence apprise (ML / DL) -> nouveaux faits
        for p in waiting:
            lvl, prob = self.ml(p)
            facts.append(("ML_Risk", p.id, lvl, prob)); self.log("ML", f"ML_Risk({p.id},{lvl},{prob})")
            dl = self.dl(p.history)
            if dl:
                facts.append(("DL_Risk", p.id, dl[0], dl[1])); self.log("DL", f"DL_Risk({p.id},{dl[0]},{dl[1]})")

        # 2) Raisonnement logique
        derived, trace = self.reasoner(facts)
        for t in trace: self.log("LOGIQUE", t)

        # 3) Exigences pour la recherche
        req = {}
        for p in waiting:
            icu = ("RequiresICU", p.id) in derived
            spec = next((f[2] for f in derived if f[0] == "RequiresSpecialist" and f[1] == p.id), None)
            prio = 3 if ("CriticalRisk", p.id) in derived else 2 if ("HighRisk", p.id) in derived else 1
            req[p.id] = {"priority": prio, "icu": icu, "specialty": spec}
        self.log("EXIGENCES", str(req))

        # 4) Recherche d'une allocation, 5) Planification
        alloc, unsat = self.searcher(self.state, req)
        self.log("RECHERCHE", f"allocation = {alloc}" + (f" | non satisfaits = {unsat}" if unsat else ""))
        self.plan = self.planner(self.state, alloc)
        self.plan_version, self.step = self.state.version, 0
        self.log("PLAN", " -> ".join(f"{a[0]}({','.join(a[1:])})" for a in self.plan))

    # ------------------------------------------------------------------ exécution pas à pas
    def plan_is_obsolete(self):
        return self.state.version != self.plan_version

    def execute_next(self):
        """Exécute UNE action du plan. Retourne False quand le plan est terminé."""
        if self.step >= len(self.plan): return False
        if self.plan_is_obsolete():
            self.log("CHANGEMENT", f"Plan obsolète (construit sur v{self.plan_version}, état actuel v{self.state.version}) -> replanification")
            self.replan(); return True
        act = self.plan[self.step]; self.step += 1
        name, pid = act[0], act[1]
        if name == "AssignDoctor": self.state.assign(pid, doctor=act[2])
        elif name in ("ReserveICUBed", "ReserveBed"): self.state.assign(pid, bed=act[2])
        elif name == "StartMonitoring": self.completed.add(pid)
        self.plan_version = self.state.version      # nos propres modifications ne rendent pas le plan obsolète
        self.log("EXEC", f"{name}({','.join(act[1:])})")
        return True

    def replan(self):
        for pid in list(self.state.assignments):
            if pid not in self.completed:
                self.state.release(pid)             # on annule les affectations non terminées
        self.cycle()

    def run_plan(self):
        while self.execute_next(): pass

    # ------------------------------------------------------------------ événements
    def handle_event(self, ev):
        self.log("EVENEMENT", str(ev))
        apply_event(self.state, ev)
