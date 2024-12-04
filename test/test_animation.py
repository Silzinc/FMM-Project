from copy import deepcopy
import vpython as vp
import numpy as np

import sys
import os

current_dir = os.path.dirname(__file__)  # Chemin du fichier actuel (test_animation.py)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Remonter au dossier parent
src_dir = os.path.join(parent_dir)  # Ajouter "src"
sys.path.append(src_dir)

import src as fmm

np.set_printoptions(precision=3, suppress=True)

size = 10.0
mu = 1.0


def gen_random_samples(n: int):
    return [fmm.MassSample((np.random.rand(3) - 0.5) * size, mass=mu) for _ in range(n)]


def phi(xi: float) -> float:
    return xi / (1 + xi * xi) ** (1 / 2)


def grad_phi(xi: float) -> float:
    return -(xi**3) / (1 + xi * xi) ** (3 / 2)


def test_animation():
    vp.scene.width = vp.scene.height = 600
    vp.scene.background = vp.color.black
    vp.scene.range = 1.3
    N = 50
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

    nb_images = 700
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
    solver = fmm.FMMSolver(size, phi, grad_phi, 0.8, deepcopy(samples), 3)

    # Générer toutes les positions des points pour les frames
    for step in range(nb_images):
        print("Calculating frame " + str(step))
        q = []
        for sample in solver.samples:
            next = sample
            q.append(vp.vec(next.pos[0], next.pos[1], next.pos[2]))
        nuage = vp.points(pos=q, size_units="world", color=vp.color.white, radius=solver.epsilon / 30, visible = False)
        p.append(nuage)
        solver.update()

    # Créer les faces des cellules

    for cell in solver.iter_cells(-1):
        assert cell is not None
        cell_to_faces(cell)


    # Animation et mise à jour dynamique des points
    while True and run:
        for frame in p:
            # Mettre à jour les positions des points avant de les rendre visibles
            frame.visible = True
            vp.rate(20)  # Vitesse de l'animation
            frame.visible = False

def main(args):
    test_animation()

if __name__ == "__main__":
    main(sys.argv)