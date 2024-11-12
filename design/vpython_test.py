import vpython as vp

vp.scene.width = vp.scene.height = 600
vp.scene.background = vp.color.white
vp.scene.range = 1.3
N = 15000
vp.scene.title = f"A {N}-element points object with random radii and colors"
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


vp.button(text="Pause", bind=Runbutton)
vp.scene.append_to_caption("""
To rotate "camera", drag with right vp.button or Ctrl-drag.
To zoom, drag with middle vp.button or Alt/Option depressed, or use scroll wheel.
  On a two-vp.button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")

p = []
last = vp.vec(0, 0, 0)
for i in range(N):
    next = last + 0.1 * vp.vec.random()
    while vp.mag(next) > 1:  # if next is outside the sphere, try another random value
        next = last + 0.1 * vp.vec.random()
    p.append(
        {
            "pos": next,
            "radius": 0.002 + 0.04 * vp.random(),
            "color": (vp.vec(1, 1, 1) + vp.vec.random()) / 2,
        }
    )
    last = next
c = vp.points(pos=p, size_units="world")
while True:
    vp.rate(60)
    if run:  # Currently there isn't a way to rotate a points object, so rotate vp.scene.forward:
        vp.scene.forward = vp.scene.forward.rotate(angle=-0.005, axis=vp.vec(0, 1, 0))
