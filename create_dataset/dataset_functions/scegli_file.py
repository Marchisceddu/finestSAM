import tkinter as tk
from tkinter import filedialog

# Open a dialog window to select a file and return the path
def get_shp_file_path():
  root = tk.Tk()
  root.withdraw()
  return filedialog.askopenfilename(title="Seleziona un file SHP", filetypes=[("Shapefile", "*.shp")])

# Open a dialog window to select a TIF file and return the path
def get_tif_file_path():
  root = tk.Tk()
  root.withdraw()
  return filedialog.askopenfilename(title="Seleziona un file TIF", filetypes=[("GeoTIFF", "*.tif")])

# Open a dialog window to select a PNG file and return the path
def get_png_file_path():
  root = tk.Tk()
  root.withdraw()
  return filedialog.askopenfilename(title="Seleziona un file PNG", filetypes=[("Portable Network Graphics", "*.png")])

# Open a dialog window to select a folder and return the path
def get_folder_path():
  root = tk.Tk()
  root.withdraw()
  return filedialog.askdirectory(title="Seleziona una cartella")