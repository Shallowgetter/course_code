import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

def sample_distribution(dist, params, n):
    if dist=="normal":
        return np.random.normal(float(params.get("mean",50)), float(params.get("std",5)), n)
    elif dist=="lognormal":
        return np.random.lognormal(float(params.get("mean",0)), float(params.get("sigma",1)), n)
    elif dist=="exponential":
        return np.random.exponential(float(params.get("scale",1)), n)
    elif dist=="weibull":
        return np.random.weibull(float(params.get("a",1)), n)*float(params.get("scale",1))
    else:
        raise ValueError("unknown distribution")

def simulate():
    try:
        n = int(sample_size_entry.get())
    except:
        messagebox.showerror("Error", "Invalid sample size")
        return
    stress_dist = stress_var.get()
    strength_dist = strength_var.get()
    stress_params = {}
    for key, entry in stress_entries.items():
        val = entry.get()
        if val=="":
            messagebox.showerror("Error", "Incomplete stress parameter: " + key)
            return
        stress_params[key] = float(val)
    strength_params = {}
    for key, entry in strength_entries.items():
        val = entry.get()
        if val=="":
            messagebox.showerror("Error", "Incomplete strength parameter: " + key)
            return
        strength_params[key] = float(val)
    stress = sample_distribution(stress_dist, stress_params, n)
    strength = sample_distribution(strength_dist, strength_params, n)
    reliability = np.mean(strength >= stress)
    messagebox.showinfo("Result", "Reliability = " + str(reliability))
    plt.hist(stress, bins=50, alpha=0.5, label="Stress")
    plt.hist(strength, bins=50, alpha=0.5, label="Strength")
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo Simulation of Reliability")
    plt.show()

def update_fields(dist, entries, frame):
    for widget in frame.winfo_children():
        widget.destroy()
    params = {"normal": ["mean", "std"], "lognormal": ["mean", "sigma"], "exponential": ["scale"], "weibull": ["a", "scale"]}
    keys = params.get(dist, [])
    for i, key in enumerate(keys):
        label = tk.Label(frame, text=key)
        label.grid(row=i, column=0)
        entry = tk.Entry(frame)
        entry.grid(row=i, column=1)
        entries[key] = entry

def update_stress_fields(*args):
    global stress_entries
    stress_entries = {}
    update_fields(stress_var.get(), stress_entries, stress_frame)

def update_strength_fields(*args):
    global strength_entries
    strength_entries = {}
    update_fields(strength_var.get(), strength_entries, strength_frame)

root = tk.Tk()
root.title("Monte Carlo Reliability Simulation")
stress_var = tk.StringVar(value="normal")
strength_var = tk.StringVar(value="normal")
frame1 = tk.Frame(root)
frame1.pack(padx=10, pady=10)
tk.Label(frame1, text="Stress Distribution").grid(row=0, column=0)
stress_menu = tk.OptionMenu(frame1, stress_var, "normal", "lognormal", "exponential", "weibull")
stress_menu.grid(row=0, column=1)
stress_frame = tk.Frame(frame1)
stress_frame.grid(row=1, column=0, columnspan=2)
tk.Label(frame1, text="Strength Distribution").grid(row=2, column=0)
strength_menu = tk.OptionMenu(frame1, strength_var, "normal", "lognormal", "exponential", "weibull")
strength_menu.grid(row=2, column=1)
strength_frame = tk.Frame(frame1)
strength_frame.grid(row=3, column=0, columnspan=2)
tk.Label(frame1, text="Sample Size").grid(row=4, column=0)
sample_size_entry = tk.Entry(frame1)
sample_size_entry.insert(0, "100000")
sample_size_entry.grid(row=4, column=1)
simulate_button = tk.Button(frame1, text="Simulate", command=simulate)
simulate_button.grid(row=5, column=0, columnspan=2, pady=10)
stress_var.trace("w", update_stress_fields)
strength_var.trace("w", update_strength_fields)
update_stress_fields()
update_strength_fields()
root.mainloop()
