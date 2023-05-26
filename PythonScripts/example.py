import numpy as np
from calc_airfoil import read_foil

continent_foil = read_foil("continent")
continent_reaf_foil = read_foil("continent_reaf")
QR_foil = read_foil("QR")
reaf_foil = read_foil("reaf")
rev_root_140_foil = read_foil("rev_root_140")
snou1013_foil = read_foil("snou1013")
QX0023_foil = read_foil("QX0023")


# np.stackを使って、これらの配列を1つの3次元配列にまとめます。
stacked_arrays = np.stack(
    [
        continent_foil,
        continent_reaf_foil,
        QR_foil,
        reaf_foil,
        rev_root_140_foil,
        snou1013_foil,
        QX0023_foil,
    ]
)

# np.meanを使って、新たな配列の平均を計算します。axis=0は第一次元（7つの配列）に沿って平均をとることを意味します。
average_array = np.mean(stacked_arrays, axis=0)

print(average_array)
foilname = "average"
f = open(foilname + ".dat", "w")
f.write(foilname)
f.write("\n")

for i in range(257):
    f.write(str(average_array[i][0]))
    f.write(" ")
    f.write(str(average_array[i][1]))
    f.write("\n")

f.close()
