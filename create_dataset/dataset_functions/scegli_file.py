import tkinter as tk
from tkinter import filedialog

def get_shp_file_path():
  """Apre una finestra di dialogo per la selezione di un file SHP e restituisce il percorso."""
  root = tk.Tk()
  root.withdraw()
  return filedialog.askopenfilename(title="Seleziona un file SHP", filetypes=[("Shapefile", "*.shp")])

def get_tif_file_path():
  """Apre una finestra di dialogo per la selezione di un file TIF e restituisce il percorso."""
  root = tk.Tk()
  root.withdraw()
  return filedialog.askopenfilename(title="Seleziona un file TIF", filetypes=[("GeoTIFF", "*.tif")])
