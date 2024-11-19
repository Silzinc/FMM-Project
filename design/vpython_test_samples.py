import vpython as vp
import fmm_solver_design as solv

vp.scene.width = vp.scene.height = 600
vp.scene.background = vp.color.black
vp.scene.range = 1.3
N = 1500
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

def cell_to_faces(cell: solv.FMMCell):
    center = vp.vec(*(cell.centroid))
    L = cell.size *0.9
    c = vp.box(pos = center, length = L, height=L, width=L, opacity= 0.1, shininess = 0, color=vp.color.cyan)

    #c = vp.cylinder(pos=origin, axis = vp.vec(L, 0, 0), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(0, L, L) , axis = vp.vec(L, 0, 0), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(0, 0, L) , axis = vp.vec(L, 0, 0), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(0, L, 0) , axis = vp.vec(L, 0, 0), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin, axis = vp.vec(0, L, 0), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(0, 0, L) , axis = vp.vec(0, L, 0), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(L, 0, 0) , axis = vp.vec(0, L, 0), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(L, 0, L) , axis = vp.vec(0, L, 0), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin, axis = vp.vec(0, 0, L), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(0, L, 0) , axis = vp.vec(0, 0, L), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(L, 0, 0) , axis = vp.vec(0, 0, L), color=vp.color.green, radius=0.01)
    #c = vp.cylinder(pos=origin + vp.vec(L, L, 0) , axis = vp.vec(0, 0, L), color=vp.color.green, radius=0.01)





vp.button(text="Pause", bind=Runbutton)
vp.scene.append_to_caption("""
To rotate "camera", drag with right vp.button or Ctrl-drag.
To zoom, drag with middle vp.button or Alt/Option depressed, or use scroll wheel.
  On a two-vp.button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")

p = []

def gen_random_samples(n: int):
    return [solv.MassSample() for _ in range(n)]


samples = gen_random_samples(N)
solver = solv.FMMSolver(10, lambda x: x, 0.1, samples, 10)

solver.make_tree()

for i in range(N):
    next = samples[i]
    p.append(
        {
            "pos": vp.vec(next.pos[0],next.pos[1],next.pos[2]),
            "radius": solver.epsilon/30,
            "color": vp.color.white,
        }
    )
    
c = vp.points(pos=p, size_units="world")

for j in range(len(solver.leaves_cells)):
    cell = solver.leaves_cells[j].value
    assert cell is not None
    cell_to_faces(cell)





while True:
    vp.rate(60)
    #if run:  # Currently there isn't a way to rotate a points object, so rotate vp.scene.forward:
    #    vp.scene.forward = vp.scene.forward.rotate(angle=-0.005, axis=vp.vec(0, 1, 0))
