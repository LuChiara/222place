# 🍽️ 222.place — Group Matching Model

> *Reverse-engineering the magic of a dinner where strangers feel like old friends.*

I've attended 5 dinners with [222.place](https://222.place). Every group felt genuinely diverse — different jobs, ages, cities — yet conversations clicked immediately. This repo is my attempt to understand why, by building the matching system myself.

---

## What's in here

| File | Description |
|---|---|
| `222_matching_model.ipynb` | Full pipeline: data → embeddings → optimisation → event schedule |
| `inputs/222_users.csv` | Synthetic dataset — 300 users, 61 attributes |
| `inputs/222_restaurants.csv` | 20 venues with capacity, cuisine, availability |
| `outputs/222_event_schedule.csv` | Generated group + event assignments |

---

## The Problem

Form groups of **6 people (3M + 3F)** from a pool of ~40 attendees per event such that:
- Pairwise compatibility within each group is maximised
- Groups are balanced by gender
- Everyone is available on the chosen day
- 5–7 groups get assigned to a single restaurant that fits their collective cuisine preferences and the venue's capacity

It sounds soft. It's actually a combinatorial optimisation problem.

---

## Architecture

### 1. Feature Engineering
60+ questionnaire attributes encoded into a normalised feature matrix:
- **Scale fields (1–7):** Big Five personality, values, lifestyle, social preferences, self-perception, traits
- **Categorical:** education, media genres (multi-hot), hometown size, relationship status
- **Logistics:** day availability, cuisine ratings per venue type

### 2. Autoencoder Embeddings
A pure-NumPy autoencoder compresses each user into a **16-dimensional embedding**:
```
Input (F-dim) → 128 → 64 → [16-dim bottleneck] → 64 → 128 → Output
```
Trained with MSE reconstruction loss, mini-batch SGD, gradient clipping.

Pairwise **cosine similarity** on embeddings gives a (300 × 300) compatibility matrix.

### 3. Group Formation — Two-Stage Optimisation

**Stage 1: Group formation (per day)**

$$\max \sum_g \sum_{i < j} c_{ij} \cdot w_{ijg}$$

Subject to:
- Each user assigned to exactly one group
- Each group has exactly 6 members (3M + 3F)
- Linearisation constraints for the $w_{ijg}$ product variables
- Day availability mask

Solved **exactly** via `scipy.optimize.milp` (HiGHS solver) for N ≤ 30.
Full scale (N = 300) uses **greedy seeding + simulated annealing**.

**Stage 2: Event assembly**

Groups are then packed into events and matched to restaurants, optimising for cuisine fit and respecting venue capacity constraints.

---

## Key Finding

> The groups that felt most *diverse* on the surface were often the highest-scoring on deep compatibility.

Different professions, ages, cities — but the same underlying openness, curiosity, and values. The surface differences made the conversation interesting. The latent similarities made it flow.

---

## Stack

```
Python 3.x · NumPy · Pandas · Scikit-learn · SciPy · Matplotlib · Seaborn
```
No deep learning frameworks needed — autoencoder is implemented in pure NumPy.

---

## Running it

```bash
git clone https://github.com/LuChiara/222place.git
cd 222place
pip install numpy pandas scikit-learn scipy matplotlib seaborn
jupyter notebook 222_matching_model.ipynb
```

---

## Caveats

This is a **synthetic** reverse-engineering exercise, not affiliated with or informed by 222.place's actual system. The dataset is generated from archetype priors, not real user data. The real magic is probably more interesting than this.

---

*Built out of curiosity after 5 dinners that left me wondering. If you're thinking about the same problems — matching, social graphs, community design or just chatting casually — feel free to open an issue or reach out.*
