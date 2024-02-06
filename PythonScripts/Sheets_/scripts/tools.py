import matplotlib.pyplot as plt


def show_data(type="wireframe", *args):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for arg in args:
        ax.plot_wireframe(
            arg[0], arg[1], arg[2]
        ) if type == "wireframe" else ax.plot_surface(
            arg[0], arg[1], arg[2]
        ) if type == "surface" else ax.plot(
            arg[0], arg[1], arg[2]
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
