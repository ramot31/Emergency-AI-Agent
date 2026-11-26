"""Bloc 1 - Modèle de l'environnement des urgences (simulation, pas d'usage médical réel)."""
from dataclasses import dataclass, field
import copy as _copy
from typing import Optional

# Seuils de discrétisation (à documenter dans le rapport)
SBP_LOW = 90        # tension systolique basse
SBP_HIGH = 140      # tension systolique haute
SPO2_LOW = 92       # saturation basse
HR_HIGH = 100       # tachycardie
TEMP_FEVER = 38.0   # fièvre


@dataclass
class Patient:
    id: str
    age: int
    heart_rate: int
    systolic_bp: int
    spo2: int
    temperature: float
    resp_rate: int = 16
    symptoms: list = field(default_factory=list)
    status: str = "waiting"          # waiting | under_treatment
    history: list = field(default_factory=list)  # mesures temporelles (pour le DL)

    def bp_level(self):
        if self.systolic_bp < SBP_LOW: return "Low"
        if self.systolic_bp > SBP_HIGH: return "High"
        return "Normal"

    def snapshot(self):
        return {"hr": self.heart_rate, "bp": self.systolic_bp, "spo2": self.spo2}


@dataclass
class Doctor:
    id: str
    specialty: str
    available: bool = True


@dataclass
class Nurse:
    id: str
    available: bool = True


@dataclass
class Bed:
    id: str
    type: str                        # "Emergency" | "ICU"
    available: bool = True


@dataclass
class Room:
    id: str
    available: bool = True


@dataclass
class Equipment:
    id: str
    kind: str
    available: bool = True


class State:
    """Etat de l'environnement à un instant donné. Chaque modification incrémente `version`."""

    def __init__(self):
        self.patients, self.doctors, self.nurses = {}, {}, {}
        self.beds, self.rooms, self.equipment = {}, {}, {}
        self.assignments = {}        # patient_id -> {"doctor": id, "bed": id, "room": id}
        self.version = 0
        self.out_of_service = set()   # ressources en panne (ne doivent pas être libérées)
        self.log = []                # historique des événements (pour la démo)

    # -------- ajout d'entités --------
    def add_patient(self, p: Patient):
        if not p.history:
            p.history.append(p.snapshot())
        self.patients[p.id] = p
        self._touch(f"Arrivée du patient {p.id}")

    def add_resource(self, kind, obj):
        {"doctor": self.doctors, "nurse": self.nurses, "bed": self.beds,
         "room": self.rooms, "equipment": self.equipment}[kind][obj.id] = obj
        self._touch(f"Ressource ajoutée : {obj.id}")

    # -------- transitions --------
    def new_measurement(self, pid, hr, bp, spo2):
        p = self.patients[pid]
        p.heart_rate, p.systolic_bp, p.spo2 = hr, bp, spo2
        p.history.append(p.snapshot())
        self._touch(f"Nouvelle mesure pour {pid}: HR={hr} BP={bp} SpO2={spo2}")

    def _resource(self, rid):
        for d in (self.doctors, self.nurses, self.beds, self.rooms, self.equipment):
            if rid in d: return d[rid]
        raise KeyError(rid)

    def set_available(self, rid, value: bool):
        self._resource(rid).available = value
        (self.out_of_service.discard if value else self.out_of_service.add)(rid)
        self._touch(f"{rid} {'disponible' if value else 'indisponible'}")

    def assign(self, pid, doctor=None, bed=None, room=None):
        a = self.assignments.setdefault(pid, {})
        for key, rid in (("doctor", doctor), ("bed", bed), ("room", room)):
            if rid:
                if not self._resource(rid).available:
                    raise ValueError(f"{rid} n'est pas disponible")
                self._resource(rid).available = False
                a[key] = rid
        self.patients[pid].status = "under_treatment"
        self._touch(f"Affectation {pid}: {a}")

    def release(self, pid):
        for rid in self.assignments.pop(pid, {}).values():
            if rid not in self.out_of_service:      # une ressource en panne reste indisponible
                self._resource(rid).available = True
        self.patients[pid].status = "waiting"
        self._touch(f"Libération des ressources de {pid}")

    def _touch(self, msg):
        self.version += 1
        self.log.append((self.version, msg))

    def copy(self):
        """Copie indépendante de l'état (utile pour la recherche : explorer sans modifier l'original)."""
        return _copy.deepcopy(self)

    # -------- requêtes utiles aux autres blocs --------
    def waiting_patients(self):
        return [p for p in self.patients.values() if p.status == "waiting"]

    def available(self, kind, **flt):
        coll = {"doctor": self.doctors, "bed": self.beds, "room": self.rooms,
                "nurse": self.nurses, "equipment": self.equipment}[kind]
        return [r for r in coll.values() if r.available and
                all(getattr(r, k) == v for k, v in flt.items())]

    def summary(self):
        return (f"[v{self.version}] {len(self.waiting_patients())} patients en attente | "
                f"médecins dispo: {[d.id for d in self.available('doctor')]} | "
                f"lits urgence dispo: {[b.id for b in self.available('bed', type='Emergency')]} | "
                f"lits USI dispo: {[b.id for b in self.available('bed', type='ICU')]}")
