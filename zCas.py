from cas import Cas
from cas_surface import CasSurface
from adjacency_matrix import *

casobject =Cas(r"Examples\big.wrl",0.001)
casobject.build_adjacency_matrix()


print("Adjacency Matrix:")
print(casobject.adjacency_matrix)

output = extract_non_adjacent_groups(casobject.adjacency_matrix)
print("Non Adjacent Domains:")
print(output)
casobject.ax.show()
pass