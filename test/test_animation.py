import os
import sys
from copy import deepcopy

import numpy as np
import vpython as vp

current_dir = os.path.dirname(__file__)  # Chemin du fichier actuel (test_animation.py)
parent_dir = os.path.abspath(
    os.path.join(current_dir, "..")
)  # Remonter au dossier parent
src_dir = os.path.join(parent_dir)  # Ajouter "src"
sys.path.append(src_dir)

import src as fmm

np.set_printoptions(precision=3, suppress=True)

size = 10.0
mu = 1.0

# Set to True to enable comparison with naive solver
naive_compare = False
# Set to True to put a high-mass sample in the center of the volume
enable_sun = True


def gen_random_samples(n: int):
    return [fmm.MassSample((np.random.rand(3) - 0.5) * size, mass=mu) for _ in range(n)]

def gen_random_clusters(n_cluster, n_sample):
    samples = []
    mean_vdt = 0
    e_z = np.array([0,0,0.6/size])
    if not enable_sun:
        e_z /= 2
    for cluster in range(n_cluster):
        center = (np.random.rand(3) - 0.5) * size * 0.7
        vdt = np.cross(center,e_z)
        mean_vdt += vdt
        for i in range(n_sample):
            pos = np.random.randn(3) + center
            last_pos = pos + vdt
            samples.append(fmm.MassSample(pos, mass=mu, prev_pos=last_pos))
    mean_vdt /= len(samples)
    for i in range(len(samples)):
        samples[i].prev_pos -= mean_vdt
    return samples


def phi(xi: float) -> float:
    return xi / (1 + xi * xi) ** (1 / 2)


def grad_phi(xi: float) -> float:
    return -(xi**3) / (1 + xi * xi) ** (3 / 2)


def test_animation():
    vp.scene.width = vp.scene.height = 600
    vp.scene.background = vp.color.black
    vp.scene.range = 1.3
    N_sample = 5
    N_cluster = 10
    vp.scene.title = f"{N_sample * N_cluster} randomly generated samples"
    # Display frames per second and render time:
    vp.scene.append_to_title("<div id='fps'/>")

    global run
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
        c = vp.box(
            pos=center,
            length=L,
            height=L,
            width=L,
            opacity=0.1,
            shininess=0,
            color=vp.color.cyan,
        )

    vp.button(text="Pause", bind=Runbutton)
    vp.scene.append_to_caption("""
    To rotate "camera", drag with right vp.button or Ctrl-drag.
    To zoom, drag with middle vp.button or Alt/Option depressed, or use scroll wheel.
    On a two-vp.button mouse, middle is left + right.
    To pan left/right and up/down, Shift-drag.
    Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")

    nb_images = 200
    frame_rate = 30
    p = []
    p_naive = []
    p_sun = []
    size = 10.0
    mu = 1.0
    dt = 0.1


    def phi(xi: fmm.Vec3) -> float:
        xin = np.linalg.norm(xi)
        return float(xin / (1 + xin * xin) ** (1 / 2))

    def grad_phi(xi: fmm.Vec3) -> fmm.Vec3:
        xin = np.linalg.norm(xi)
        return -(xin**2) / (1 + xin * xin) ** (3 / 2) * xi

    # Pour le solver d'ordre 1, ajouter hess_phi en dernier argument de FMMSolver
    def hess_phi(xi: fmm.Vec3) -> fmm.Mat3x3:
        xin = np.linalg.norm(xi)
        return xin**3 * (
            3 * np.outer(xi, xi) / (1 + xin**2) ** (5 / 2)
            - np.eye(3) / (1 + xin**2) ** (3 / 2)
        )

    def gen_random_samples(n: int):
        return [
            fmm.MassSample((np.random.rand(3) - 0.5) * size * 0.7, mass=mu) for _ in range(n)
        ]

    samples = gen_random_clusters(N_cluster, N_sample)
    if enable_sun:
        samples.append(fmm.MassSample(np.array([0,0,0.01]), mass=3000*mu)) # adding a sun
    solver = fmm.FMMSolver(size, dt, deepcopy(samples), 3, phi, grad_phi, hess_phi) # add hess_phi as arg for 1st order
    epsilon = 4 * size / np.sqrt(len(samples))
    naive_solver = fmm.NaiveSolver(size,dt,epsilon,deepcopy(samples), phi, grad_phi)
    

    # Générer toutes les positions des points pour les frames
    for step in range(nb_images):
        print("Calculating frame " + str(step))
        q = []
        q_naive = []
        for sample in solver.samples:
            next = sample
            q.append(vp.vec(next.pos[0], next.pos[1], next.pos[2]))
        nuage = vp.points(
            pos=q,
            size_units="world",
            color=vp.color.white,
            radius=solver.epsilon / 60,
            visible=False,
        )
        p.append(nuage)
        if enable_sun:
            sun = solver.samples[-1]
            p_sun.append(vp.sphere(
                pos=vp.vec(sun.pos[0], sun.pos[1], sun.pos[2]),
                size_units="world",
                color=vp.color.orange,
                radius=solver.epsilon / 30,
                visible=False,
            ))
        solver.update()

        # Takes place only if naive solver is also required for comparison
        if naive_compare:
            for sample in naive_solver.samples:
                next = sample
                q_naive.append(vp.vec(next.pos[0], next.pos[1], next.pos[2]))
            naive_nuage = vp.points(
                pos=q_naive,
                size_units="world",
                color=vp.color.yellow,
                radius=solver.epsilon / 60, # Warning, same size as the FMMsolver for graphic purpose
                visible=False,
            )
            p_naive.append(naive_nuage)
            naive_solver.update()

    # Créer les faces des cellules

    for cell in solver.iter_cells(-1):
        assert cell is not None
        cell_to_faces(cell)

    # Animation et mise à jour dynamique des points
    image = 0
    while True:
        if run:
            vp.rate(frame_rate)  # Vitesse de l'animation
            frame = p[image]
            frame.visible = False
            if enable_sun:
                sun_frame = p_sun[image]            
                sun_frame.visible = False
            if naive_compare:
                naive_frame = p_naive[image]
                naive_frame.visible = False
            # Mettre à jour les positions des points avant de les rendre visibles
            image = (image+1)%len(p)
            frame = p[image]
            frame.visible = True
            if enable_sun:
                sun_frame = p_sun[image]            
                sun_frame.visible = True
            if naive_compare:
                naive_frame = p_naive[image]
                naive_frame.visible = True


def main(args):
    test_animation()


if __name__ == "__main__":
    main(sys.argv)
