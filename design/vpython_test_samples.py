import vpython as vp
from copy import deepcopy

import numpy as np
import src as fmm

vp.scene.width = vp.scene.height = 600
vp.scene.background = vp.color.black
vp.scene.range = 1.3
N = 900
vp.scene.title = f"{N} randomly generated samples"
# Display frames per second and render time:
vp.scene.append_to_title("<div id='fps'/>")

run = True

def Runbutton(b):
    global run
    if b.text == "Pause":
        run = False
        b.text = "Run"
    else:
        run = True
        b.text = "Pause"

def cell_to_faces(cell: fmm.FMMCell):
    center = vp.vec(*(cell.centroid))
    L = cell.size * 0.9
    c = vp.box(pos=center, length=L, height=L, width=L, opacity=0.1, shininess=0, color=vp.color.cyan)

vp.button(text="Pause", bind=Runbutton)
vp.scene.append_to_caption("""
To rotate "camera", drag with right vp.button or Ctrl-drag.
To zoom, drag with middle vp.button or Alt/Option depressed, or use scroll wheel.
  On a two-vp.button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")

nb_images = 100
p = []
size = 10.0
mu = 1.0

def phi(xi: float) -> float:
    return xi / (1 + xi * xi) ** (1 / 2)

def grad_phi(xi: float) -> float:
    return -(xi**3) / (1 + xi * xi) ** (3 / 2)

def gen_random_samples(n: int):
    return [fmm.MassSample((np.random.rand(3) - 0.5) * size, mass=mu) for _ in range(n)]

samples = gen_random_samples(N)
solver = fmm.FMMSolver(size, phi, grad_phi, 0.1, deepcopy(samples), 3)

# Générer toutes les positions des points pour les frames
for step in range(nb_images):
    print("Calculating frame " + str(step))
    q = []
    for sample in solver.samples:
        next = sample
        q.append(vp.vec(next.pos[0], next.pos[1], next.pos[2]))
    p.append(q)
    solver.update()

# Créer les faces des cellules

for cell in solver.iter_cells(-1):
    assert cell is not None
    cell_to_faces(cell)

# Création des objets points avec toutes les positions initiales
c = vp.points(pos=p[0], size_units="world", color=vp.color.white, radius=solver.epsilon/30)

# Animation et mise à jour dynamique des points
while True and run:
    for frame in range(nb_images):
        # Mettre à jour les positions des points avant de les rendre visibles
        vp.points(pos=p[0], size_units="world", color=vp.color.white, radius=solver.epsilon/30)
        vp.rate(30)  # Vitesse de l'animation
