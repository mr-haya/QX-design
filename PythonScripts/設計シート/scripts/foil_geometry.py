import matplotlib.pyplot as plt


def thickness_at(dat, at):
    """
    任意位置での翼型の厚さを返す関数
    """
    thickness = dat["y"].max() - dat["y"].min()
    return thickness


def max_thickness(dat):
    """
    翼型の最大厚さを返す関数
    """
    max_thickness = dat["y"].max() - dat["y"].min()
    return max_thickness


def camber_at(dat, at):
    """
    任意位置での翼型のキャンバーを返す関数
    """
    camber = dat["x"].max() - dat["x"].min()
    return camber


def max_camber(dat):
    """
    翼型の最大キャンバーを返す関数
    """
    max_camber = dat["x"].max() - dat["x"].min()
    return max_camber


def leading_edge_radius(dat):
    """
    翼型の前縁半径を返す関数
    """
    leading_edge_radius = 0.1
    return leading_edge_radius


def trailing_edge_angle(dat):
    """
    翼型の後縁角を返す関数
    """
    trairing_edge_angle = 0.1
    return trairing_edge_angle


def draw_outline(foil_name):
    dpi = 72  # 画像の解像度
    figsize = (10, 2)  # 画像のサイズ

    foil_data = foil_fetch.dat(foil_name)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect="equal")
    ax.plot([r[0] for r in foil_data], [r[1] for r in foil_data], label="original")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig
