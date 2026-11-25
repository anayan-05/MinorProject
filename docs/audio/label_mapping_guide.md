# PathPilot Audio ML — Acoustic Label Mapping & Taxonomy Guide
**Owner:** Debajyoti (Audio ML Lead)  
**Version:** 1.0.0 | **Updated:** November 2026  

---

## 1. The Core Scientific Philosophy: Physical Acoustics vs. Human State
In public safety AI, a critical error is confusing an **acoustic sound event** (a measurable physical vibration in the air) with an **interpreted human situation** (crime, danger, assault, or distress).

PathPilot takes an uncompromising stance:
> **The audio model detects objective sound categories only. It never infers assault, danger, distress, or crime.**

```text
[ Physical Sound Waves ] ──> [ Acoustic Feature Extraction ] ──> [ Physical Category: Siren / Scream / Crash ]
                                                                             │
                                                                             ▼ (NEVER AUTOMATIC)
                                                                 [ Subjective Claim: "Assault" ] ❌ FORBIDDEN
```

---

## 2. Why "Assault Detection" and "Distress Detection" Are Rejected

### A. The Absence of Ground Truth in Acoustic Physics
There is no unique acoustic frequency spectrum or mathematical signature that corresponds to "assault" or "crime".
- A loud human shout could be a joyful cheer during an IPL cricket match at Eden Gardens, a vendor calling out street food on Park Street, an actor rehearsing a line, or genuine fright.
- Breaking glass could be a busboy dropping a tumbler in a busy Dhaba or an act of vandalism.
- A siren could be an ambulance stuck in traffic $500\text{ metres}$ away, not a direct emergency affecting the traveller.

### B. The Peril of Catastrophic False Positives
If an AI system labels audio as "Assault Detected" or automatically triggers SOS:
1. It strips personal agency and autonomy from the woman traveller.
2. It can trigger panicked phone calls to family members or false alarms to police while the user is simply in a loud restaurant or market.
3. It induces acute psychological panic in the traveller when false alarms flash on screen.

### C. The Responsible Engineering Solution
PathPilot treats audio cues as **purely advisory acoustic indicators**:
1. High-frequency scream/shout $\rightarrow$ The model outputs `screaming_shouting` with a confidence score.
2. The Android client app displays an unobtrusive advisory prompt:  
   *"PathPilot noticed loud shouting nearby. Are you doing okay?"*
3. The traveller retains $100\%$ control to confirm safety, open the safe routing map, view nearby open pharmacies, or tap SOS if needed.

---

## 3. Target 5-Class Taxonomy

| Class ID | Target Label | Scope & Description | Excluded / Rejected Interpretations |
|---|---|---|---|
| `0` | `background_normal` | Ambient urban environment: traffic hum, car engines idling, human speech, laughter, footsteps, quiet streets, wind, rain. | Cannot assume absence of danger; simply indicates no acute acoustic anomaly. |
| `1` | `screaming_shouting` | High-energy human vocalizations: loud shouting, shrieking, screams, aggressive vocal calls. | Does NOT mean "assault" or "fear". |
| `2` | `alarm_siren` | Auditory warning devices: ambulance/police sirens, car alarms, continuous emergency horns. | Does NOT confirm proximity of police assistance. |
| `3` | `glass_breaking` | High-frequency shattering and impact sounds of glass panes, windowpanes, bottles. | Does NOT imply vandalism or forced entry. |
| `4` | `impact_crash` | High-energy blunt mechanical impacts: vehicular collisions, metal crushing, heavy structural impact. | Does NOT imply injury or vehicular accident involvement. |

---

## 4. Source Dataset Mapping Rules

### ESC-50 Mappings
| ESC-50 Original Class | Target Mapped Class | Decision Rationale |
|---|---|---|
| `screaming` | `screaming_shouting` | Direct acoustic match for vocal scream. |
| `siren` | `alarm_siren` | Direct match for emergency vehicle siren. |
| `car_horn` | `alarm_siren` | Audible alert acoustic cue. |
| `glass_breaking` | `glass_breaking` | Direct acoustic match for shattering glass. |
| `engine`, `train`, `airplane` | `background_normal` | Normal ambient transportation background. |
| `laughter`, `footsteps` | `background_normal` | Benign human acoustic baseline. |
| `dog`, `rooster`, `rain` | `background_normal` | Common ambient environmental sounds. |

### UrbanSound8K Mappings
| UrbanSound8K Class | Target Mapped Class | Decision Rationale |
|---|---|---|
| `siren` | `alarm_siren` | Direct match. |
| `car_horn` | `alarm_siren` | Direct match. |
| `air_conditioner` | `background_normal` | Stationary urban mechanical background. |
| `children_playing` | `background_normal` | Benign human activity. |
| `drilling`, `jackhammer` | `background_normal` | Urban construction noise. |
| `street_music` | `background_normal` | Benign musical acoustic background. |
| `gun_shot` | **EXCLUDED** | Outside scope of urban transit acoustic taxonomy in Kolkata context. |

### FSD50K Mappings
| FSD50K Sound Tag | Target Mapped Class | Decision Rationale |
|---|---|---|
| `Screaming`, `Yell`, `Shout` | `screaming_shouting` | Vocal acoustic group. |
| `Siren`, `Civil_defense_siren` | `alarm_siren` | Emergency alert devices. |
| `Shatter`, `Glass` | `glass_breaking` | High-frequency impact. |
| `Crash`, `Car_crash` | `impact_crash` | Mechanical collision. |
| `Traffic_noise`, `Speech`, `Walk` | `background_normal` | Baseline urban acoustic environment. |
