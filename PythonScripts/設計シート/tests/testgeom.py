# 追記部分
import matplotlib.pyplot as plt
import ezdxf
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# 作図部分
doc = ezdxf.new("R2010")
msp = doc.modelspace()
msp.add_line((0, 0), (10, 0))
doc.saveas("図面1.dxf")  # dxfの保存　確認だけなら不要
# 追記部分
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ctx = RenderContext(doc)
out = MatplotlibBackend(ax)
Frontend(ctx, out).draw_layout(msp, finalize=True)
plt.show()
