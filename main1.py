import tkinter as tk
import subprocess

def dehaze_program():
    #give path to your dehaze_gui.py file
    subprocess.run(["python", r"C:\Users\HP\Desktop\FinalProjectFolderWithReferences[1]\FinalProjectFolder\MainCode\dehaze_gui.py"])

def hd_program():
    #give path to your hd_gui.py file 
    subprocess.run(["python", r"C:\Users\HP\Desktop\FinalProjectFolderWithReferences[1]\FinalProjectFolder\MainCode\hd_gui.py"])

# Create the main window
root = tk.Tk()
root.geometry("600x250")

# Set the name of the window
root.title("Enhancing Safety and Clear Vision")

# Create a label for the title line
title_label = tk.Label(root, text="Intelligent Dehazing and Human Detection Solution", font=("Helvetica", 16))
title_label.pack(pady=10)

# Create buttons for dehaze and hd programs
button_dehaze = tk.Button(root, text="Dehaze Program", command=dehaze_program)
button_dehaze.pack(pady=10)

button_hd = tk.Button(root, text="HD Program", command=hd_program)
button_hd.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
