import pyvista
from pyvista import examples

pl = pyvista.Plotter()
pl.import_vrml("CAS_2_domains2.wrl")
pl.export_vrml()
pl.show()