import numpy as np
import matplotlib.pyplot as plt

# Define circle center and radius
center = (1, 1)
radius = 1

# Define four vertices of the square
square_points = np.array([
    [3, 2],
    [6, 2],
    [3, 5],
    [6, 5]
])

vertex_features = []
# Calculate distance from each vertex to the circle
for vertex in square_points:
    dist = np.linalg.norm(vertex - np.array(center)) - radius
    vertex_features.append(dist)
vertex_directions = []
for vertex in square_points:
    direction = (vertex - np.array(center)) / np.linalg.norm(vertex - np.array(center))
    vertex_directions.append(direction)

def sample_points(square_points):
    min_x, min_y = np.min(square_points, axis=0)
    max_x, max_y = np.max(square_points, axis=0)
    xs = np.linspace(min_x, max_x, 64)
    ys = np.linspace(min_y, max_y, 64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x, grid_y], axis=-1)

def extra_interpolation(points, square_points, vertex_features, vertex_directions):
    extra_interpolated_sdf = []
    for i in range(4):
        # Calculate distance between each sampling point and vertex
        offsets = points - square_points[i]
        # Use dot product to calculate projection in direction (preserve sign)
        lengths = np.sum(offsets * vertex_directions[i], axis=-1)
        # Use broadcasting to convert scalar + (64,64) to (64,64), then accumulate and average
        extra_interpolated_sdf.append(vertex_features[i] + lengths)
    return extra_interpolated_sdf

def linear_interpolation(points, extra_interpolated_sdf, square_points):
    # Dynamically calculate rectangle size
    min_corner = np.min(square_points, axis=0)
    max_corner = np.max(square_points, axis=0)
    size = max_corner - min_corner
    offsets = (points - min_corner) / size
    u = offsets[:, :, 0]
    v = offsets[:, :, 1]
    f00 = extra_interpolated_sdf[0]
    f10 = extra_interpolated_sdf[1]
    f01 = extra_interpolated_sdf[2]
    f11 = extra_interpolated_sdf[3]
    # Interpolate along x direction
    f0 = (1 - u) * f00 + u * f10
    f1 = (1 - u) * f01 + u * f11
    
    # Interpolate along y direction
    return (1 - v) * f0 + v * f1

def calculate_sdf(points, center, radius):
    return np.linalg.norm(points - center, axis=-1) - radius

if __name__ == "__main__":
    points = sample_points(square_points)
    gt_sdf = calculate_sdf(points, center, radius)
    # Test the fixed function
    extra_interpolated_sdf = extra_interpolation(points, square_points, vertex_features, vertex_directions)
    extra_interpolated_sdf_result = np.mean(np.stack(extra_interpolated_sdf, axis=-1), axis=-1)
    linear_interpolated_sdf_result = linear_interpolation(points, extra_interpolated_sdf, square_points)
    error_extra_interpolated = extra_interpolated_sdf_result - gt_sdf
    error_linear_interpolated = linear_interpolated_sdf_result - gt_sdf
    loss_extra_interpolated = np.mean(np.abs(error_extra_interpolated))
    loss_linear_interpolated = np.mean(np.abs(error_linear_interpolated))
    print(f"loss_extra_interpolated: {loss_extra_interpolated}")
    print(f"loss_linear_interpolated: {loss_linear_interpolated}")
    
    # Visualize results
    plt.figure(figsize=(20, 8))
    
    # First subplot: Extra Interpolation results
    plt.subplot(1, 2, 1)
    
    # Dynamically calculate boundaries
    min_x, min_y = np.min(square_points, axis=0)
    max_x, max_y = np.max(square_points, axis=0)
    
    # Display interpolation results (only in square region)
    plt.imshow(extra_interpolated_sdf_result, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='viridis', alpha=0.7)
    plt.colorbar(label='Extra Interpolated SDF')
    
    # Draw circle
    circle1 = plt.Circle(center, radius, fill=False, color='blue', linewidth=2, label='Circle')
    plt.gca().add_patch(circle1)
    
    # Draw square boundary
    square_x = [min_x, max_x, max_x, min_x, min_x]  # closed square
    square_y = [min_y, min_y, max_y, max_y, min_y]
    plt.plot(square_x, square_y, 'r-', linewidth=2, label='Square Boundary')
    
    # Plot square vertices
    plt.scatter(square_points[:, 0], square_points[:, 1], c='red', s=100, marker='o', 
                edgecolors='black', linewidth=1, label='Square Vertices', zorder=5)
    
    # Plot circle center
    plt.scatter(center[0], center[1], c='blue', s=100, marker='x', 
                linewidth=3, label='Circle Center', zorder=5)
    
    # Set axis limits to show all elements
    plt.xlim(0, 6)
    plt.ylim(0, 5)
    
    # Set grid and axis labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Extra Interpolation (Loss: {loss_extra_interpolated:.4f})', fontsize=14)
    
    # Ensure equal axis proportions
    plt.axis('equal')
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Add axis annotations
    plt.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # Second subplot: Linear Interpolation results
    plt.subplot(1, 2, 2)
    
    # Display interpolation results (only in square region)
    plt.imshow(linear_interpolated_sdf_result, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='viridis', alpha=0.7)
    plt.colorbar(label='Linear Interpolated SDF')
    
    # Draw circle
    circle2 = plt.Circle(center, radius, fill=False, color='blue', linewidth=2, label='Circle')
    plt.gca().add_patch(circle2)
    
    # Draw square boundary
    plt.plot(square_x, square_y, 'r-', linewidth=2, label='Square Boundary')
    
    # Plot square vertices
    plt.scatter(square_points[:, 0], square_points[:, 1], c='red', s=100, marker='o', 
                edgecolors='black', linewidth=1, label='Square Vertices', zorder=5)
    
    # Plot circle center
    plt.scatter(center[0], center[1], c='blue', s=100, marker='x', 
                linewidth=3, label='Circle Center', zorder=5)
    
    # Set axis limits to show all elements
    plt.xlim(0, 6)
    plt.ylim(0, 5)
    
    # Set grid and axis labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Linear Interpolation (Loss: {loss_linear_interpolated:.4f})', fontsize=14)
    
    # Ensure equal axis proportions
    plt.axis('equal')
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Add axis annotations
    plt.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.show()  # Display interpolation results
    
    # Create error visualization chart
    plt.figure(figsize=(20, 8))
    
    # First subplot: Extra Interpolation error
    plt.subplot(1, 2, 1)
    
    im1 = plt.imshow(error_extra_interpolated, extent=[min_x, max_x, min_y, max_y], 
                     origin='lower', cmap='RdBu_r', alpha=0.8)
    plt.colorbar(im1, label='Error (Predicted - Ground Truth)')
    
    # Draw circle
    circle_err1 = plt.Circle(center, radius, fill=False, color='black', linewidth=2, label='Circle')
    plt.gca().add_patch(circle_err1)
    
    # Draw square boundary
    plt.plot(square_x, square_y, 'k-', linewidth=2, label='Square Boundary')
    
    # Plot square vertices
    plt.scatter(square_points[:, 0], square_points[:, 1], c='black', s=100, marker='o', 
                edgecolors='white', linewidth=1, label='Square Vertices', zorder=5)
    
    # Plot circle center
    plt.scatter(center[0], center[1], c='black', s=100, marker='x', 
                linewidth=3, label='Circle Center', zorder=5)
    
    # Set axis limits to show all elements
    plt.xlim(0, 7)
    plt.ylim(0, 6)
    
    # Set grid and axis labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Extra Interpolation Error (MAE: {loss_extra_interpolated:.4f})', fontsize=14)
    
    # Ensure equal axis proportions
    plt.axis('equal')
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Second subplot: Linear Interpolation error
    plt.subplot(1, 2, 2)
    
    im2 = plt.imshow(error_linear_interpolated, extent=[min_x, max_x, min_y, max_y], 
                     origin='lower', cmap='RdBu_r', alpha=0.8)
    plt.colorbar(im2, label='Error (Predicted - Ground Truth)')
    
    # Draw circle
    circle_err2 = plt.Circle(center, radius, fill=False, color='black', linewidth=2, label='Circle')
    plt.gca().add_patch(circle_err2)
    
    # Draw square boundary
    plt.plot(square_x, square_y, 'k-', linewidth=2, label='Square Boundary')
    
    # Plot square vertices
    plt.scatter(square_points[:, 0], square_points[:, 1], c='black', s=100, marker='o', 
                edgecolors='white', linewidth=1, label='Square Vertices', zorder=5)
    
    # Plot circle center
    plt.scatter(center[0], center[1], c='black', s=100, marker='x', 
                linewidth=3, label='Circle Center', zorder=5)
    
    # Set axis limits to show all elements
    plt.xlim(0, 7)
    plt.ylim(0, 6)
    
    # Set grid and axis labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Linear Interpolation Error (MAE: {loss_linear_interpolated:.4f})', fontsize=14)
    
    # Ensure equal axis proportions
    plt.axis('equal')
    
    # Add legend
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    
    print("Error analysis completed!")
    print("Interpolation results and error charts will be displayed, close chart windows to continue program...")
    plt.show()  # Blocking display until manually close window
    print(f"Result matrix shape: {extra_interpolated_sdf_result.shape}")
    print(f"Value range: [{extra_interpolated_sdf_result.min():.3f}, {extra_interpolated_sdf_result.max():.3f}]")
    print(f"Circle center: {center}")
    print(f"Circle radius: {radius}")
    print(f"Square vertices:")
    for i, vertex in enumerate(square_points):
        print(f"  Vertex {i+1}: ({vertex[0]}, {vertex[1]})")


