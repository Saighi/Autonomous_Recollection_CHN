#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

sns.set_style("darkgrid") 
sns.set_context("paper", font_scale=1.5)

def plot_dotproduct_interpolate_plane(filename, scale):
    """
    Plots a quiver + stream field in the 2D plane spanned (affinely) by:
      - pattern_1_rate → fully_inactivated
      - pattern_2_rate → fully_activated
    """
    # 1) Load data
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]
    
    # 2) Determine grid size N
    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N*N != total_points:
        raise ValueError(f"Data has {total_points} rows, not an NxN grid.")

    # 4) Reshape for NxN
    X = x.reshape(N, N).T
    Y = y.reshape(N, N).T
    DX = dx.reshape(N, N).T
    DY = dy.reshape(N, N).T

    # 5) Ensure ascending Y
    if not np.all(np.diff(Y[:,0]) > 0):
        X = np.flipud(X)
        Y = np.flipud(Y)
        DX = np.flipud(DX)
        DY = np.flipud(DY)

    # 6) Build subplots
    fig, ax0 = plt.subplots(1, 1, figsize=(6, 6))

    # Create evenly spaced grid for streamplot
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    
    # Create new grid with evenly spaced points
    xi = np.linspace(x_min, x_max, N)
    yi = np.linspace(y_min, y_max, N)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate DX and DY onto this new grid
    points = np.column_stack((X.flatten(), Y.flatten()))
    DXI = griddata(points, DX.flatten(), (XI, YI), method='cubic')
    DYI = griddata(points, DY.flatten(), (XI, YI), method='cubic')
    
    # Stream plot with interpolated data
    strm = ax0.streamplot(xi, yi, DXI, DYI, density=0.8, color='tab:blue', arrowsize=1.7)
    ax0.set_xlabel(r"$\lambda_1$")
    ax0.set_ylabel(r"$\lambda_2$")

    # Mark pattern positions
    ax0.plot(0, 0, 'o', markersize=10, c="red") 
    ax0.plot(1, 0, 'o', markersize=4.5, c="red") 
    ax0.plot(0, 1, 'o', markersize=4.5, c="red") 

    ax0.text(0.96, -0.08, r"$\mathbf{1}$", c="red", fontsize=15)
    ax0.text(-0.015, 1.03, r"$\mathbf{2}$", c="red", fontsize=15)
    
    plt.title("Vector Field")
    
    return fig, ax0

def plot_energy_landscape(filename):
    """
    Plots the energy landscape in the 2D plane spanned by two patterns.
    """
    # 1) Load data
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    energy = data[:, 2]
    
    # 2) Determine grid size N
    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N*N != total_points:
        raise ValueError(f"Data has {total_points} rows, not an NxN grid.")

    # 3) Reshape for NxN
    X = x.reshape(N, N).T
    Y = y.reshape(N, N).T
    E = energy.reshape(N, N).T
    
    # 4) Ensure ascending Y
    if not np.all(np.diff(Y[:,0]) > 0):
        X = np.flipud(X)
        Y = np.flipud(Y)
        E = np.flipud(E)
    
    # 5) Create figure
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    
    # 6) Create contour plot with colorbar
    vmin = np.min(E)
    vmax = np.max(E)
    
    # Create contour plot
    contour = ax.contourf(X, Y, E, 20, cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Add contour lines
    contour_lines = ax.contour(X, Y, E, 8, colors='white', alpha=0.5, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Energy')
    
    # Set labels and title
    ax.set_xlabel(r"$\lambda_1$")
    ax.set_ylabel(r"$\lambda_2$")
    plt.title("Energy Landscape")
    
    # Mark pattern positions
    ax.plot(0, 0, 'o', markersize=10, c="red") 
    ax.plot(1, 0, 'o', markersize=4.5, c="red") 
    ax.plot(0, 1, 'o', markersize=4.5, c="red") 

    ax.text(0.96, -0.08, r"$\mathbf{1}$", c="red", fontsize=15)
    ax.text(-0.015, 1.03, r"$\mathbf{2}$", c="red", fontsize=15)
    
    return fig, ax

def plot_combined_landscape(vector_field_file, energy_field_file):
    """
    Creates a side-by-side plot showing both vector field and energy landscape.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1) Load vector field data
    data = np.loadtxt(vector_field_file)
    x = data[:, 0]
    y = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]
    
    # 2) Load energy data
    energy_data = np.loadtxt(energy_field_file)
    ex = energy_data[:, 0]
    ey = energy_data[:, 1]
    energy = energy_data[:, 2]
    
    # 3) Determine grid size N
    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N*N != total_points:
        raise ValueError(f"Data has {total_points} rows, not an NxN grid.")
    
    # 4) Reshape for NxN
    X = x.reshape(N, N).T
    Y = y.reshape(N, N).T
    DX = dx.reshape(N, N).T
    DY = dy.reshape(N, N).T
    
    EX = ex.reshape(N, N).T
    EY = ey.reshape(N, N).T
    E = energy.reshape(N, N).T
    
    # 5) Ensure ascending Y
    if not np.all(np.diff(Y[:,0]) > 0):
        X = np.flipud(X)
        Y = np.flipud(Y)
        DX = np.flipud(DX)
        DY = np.flipud(DY)
        
    if not np.all(np.diff(EY[:,0]) > 0):
        EX = np.flipud(EX)
        EY = np.flipud(EY)
        E = np.flipud(E)
    
    # 6) Plot vector field - Using quiver instead of streamplot for irregular grid
    # Create evenly spaced grid for streamplot
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    
    # Create new grid with evenly spaced points
    xi = np.linspace(x_min, x_max, N)
    yi = np.linspace(y_min, y_max, N)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate DX and DY onto this new grid
    points = np.column_stack((X.flatten(), Y.flatten()))
    DXI = griddata(points, DX.flatten(), (XI, YI), method='cubic')
    DYI = griddata(points, DY.flatten(), (XI, YI), method='cubic')
    
    # Stream plot with interpolated data
    strm = ax1.streamplot(xi, yi, DXI, DYI, density=0.8, color='tab:blue', arrowsize=1.7)
    ax1.set_xlabel(r"$\lambda_1$")
    ax1.set_ylabel(r"$\lambda_2$")
    ax1.set_title("Vector Field")
    
    # Mark pattern positions on vector field
    ax1.plot(0, 0, 'o', markersize=10, c="red") 
    ax1.plot(1, 0, 'o', markersize=4.5, c="red") 
    ax1.plot(0, 1, 'o', markersize=4.5, c="red") 
    ax1.text(0.96, -0.08, r"$\mathbf{1}$", c="red", fontsize=15)
    ax1.text(-0.015, 1.03, r"$\mathbf{2}$", c="red", fontsize=15)
    
    # 7) Plot energy landscape
    vmin = np.min(E)
    vmax = np.max(E)
    contour = ax2.contourf(EX, EY, E, 60, cmap='viridis', vmin=vmin, vmax=vmax,)
    contour_lines = ax2.contour(EX, EY, E, 8, colors='white', alpha=0.5, linewidths=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax2)
    cbar.set_label('Energy')
    
    ax2.set_xlabel(r"$\lambda_1$")
    ax2.set_ylabel(r"$\lambda_2$")
    ax2.set_title("Energy Landscape")
    
    # Mark pattern positions on energy landscape
    ax2.plot(0, 0, 'o', markersize=10, c="red") 
    ax2.plot(1, 0, 'o', markersize=4.5, c="red") 
    ax2.plot(0, 1, 'o', markersize=4.5, c="red") 
    ax2.text(0.96, -0.08, r"$\mathbf{1}$", c="red", fontsize=15)
    ax2.text(-0.015, 1.03, r"$\mathbf{2}$", c="red", fontsize=15)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_all_stages(folder, stages):
    """
    Create plots for all stages of the simulation process.
    
    Parameters:
    -----------
    folder : str
        Path to the simulation folder containing data files
    stages : list of str
        List of stages to plot (e.g., ["pre_train", "post_train"])
    """
    for stage in stages:
        vector_field_file = f"{folder}/vector_field_two_patterns_{stage}.txt"
        energy_field_file = f"{folder}/energy_field_two_patterns_{stage}.txt"
        
        try:
            # Plot combined landscape
            fig, _ = plot_energy_landscape(energy_field_file)
            fig.suptitle(f"Stage: {stage}", fontsize=16)
            plt.show()
        except Exception as e:
            print(f"Error plotting {stage}: {e}")

#%%
# Example usage
if __name__ == "__main__":
    folder = "../../data/all_data_splited/trained_networks_fast/Fig_patterns_vector_field_energy/sim_nb_0/"
    
    # List of stages to plot
    stages = [
        "pre_train",
        "post_train", 
        "post_inhib",
        "post_inhib_2",
        "post_null_w"
    ]
    
    # Plot all stages
    plot_all_stages(folder, stages)
# %%
