import numpy as np
import math
from typing import Tuple, Optional, List
from tinyUtilities import barycentric_coords, interpolate_height, face_slope, aspect, point_in_triangle


class GeometricSearchQueries:
    """
    Fast geometric search queries for 2.5D TIN (Triangulated Irregular Network).
    Implements elevation queries, nearest surface point finding, and shortest path algorithms.
    """
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, adjacency: np.ndarray):
        """
        Initialize with TIN data.
        
        Args:
            vertices: Array of shape (n, 3) with x,y,z coordinates
            faces: Array of shape (m, 3) with triangle vertex indices
            adjacency: Array of shape (m, 3) with face adjacency information
        """
        self.vertices = vertices
        self.faces = faces
        self.adjacency = adjacency
        
        # Build uniform grid for fast triangle lookup
        self._build_uniform_grid()
        
        # Precompute triangle properties
        self._precompute_triangle_properties()
    
    def _build_uniform_grid(self):
        """Build uniform grid binning over XY for fast triangle lookup."""
        if len(self.vertices) == 0:
            self.grid = None
            self.grid_bounds = None
            return
            
        # Get XY bounds
        x_min, x_max = self.vertices[:, 0].min(), self.vertices[:, 0].max()
        y_min, y_max = self.vertices[:, 1].min(), self.vertices[:, 1].max()
        
        # Determine grid resolution (aim for ~10 triangles per cell)
        n_triangles = len(self.faces)
        grid_size = max(1, int(math.sqrt(n_triangles / 10)))
        
        self.grid_bounds = (x_min, x_max, y_min, y_max)
        self.grid_size = grid_size
        
        # Create grid
        self.grid = [[] for _ in range(grid_size * grid_size)]
        
        # Assign triangles to grid cells
        for face_idx, face in enumerate(self.faces):
            if len(face) == 3:
                # Get triangle bounding box
                v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
                tri_x_min = min(v0[0], v1[0], v2[0])
                tri_x_max = max(v0[0], v1[0], v2[0])
                tri_y_min = min(v0[1], v1[1], v2[1])
                tri_y_max = max(v0[1], v1[1], v2[1])
                
                # Find grid cells that overlap with triangle bounding box
                cell_x_min = max(0, int((tri_x_min - x_min) / (x_max - x_min) * grid_size))
                cell_x_max = min(grid_size - 1, int((tri_x_max - x_min) / (x_max - x_min) * grid_size))
                cell_y_min = max(0, int((tri_y_min - y_min) / (y_max - y_min) * grid_size))
                cell_y_max = min(grid_size - 1, int((tri_y_max - y_min) / (y_max - y_min) * grid_size))
                
                for cell_x in range(cell_x_min, cell_x_max + 1):
                    for cell_y in range(cell_y_min, cell_y_max + 1):
                        cell_idx = cell_y * grid_size + cell_x
                        self.grid[cell_idx].append(face_idx)
    
    def _precompute_triangle_properties(self):
        """Precompute triangle normals, slopes, and aspects."""
        self.triangle_normals = []
        self.triangle_slopes = []
        self.triangle_aspects = []
        
        for face in self.faces:
            if len(face) == 3:
                v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
                
                # Calculate normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)  # Normalize
                
                self.triangle_normals.append(normal)
                self.triangle_slopes.append(face_slope(normal))
                self.triangle_aspects.append(aspect(normal))
            else:
                self.triangle_normals.append(np.array([0, 0, 1]))
                self.triangle_slopes.append(0)
                self.triangle_aspects.append(0)
    
    def _get_candidate_triangles(self, x: float, y: float) -> List[int]:
        """Get candidate triangles for a given (x,y) point using uniform grid."""
        if self.grid is None:
            return []
            
        x_min, x_max, y_min, y_max = self.grid_bounds
        
        # Check if point is within bounds
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return []
        
        # Get grid cell
        cell_x = int((x - x_min) / (x_max - x_min) * self.grid_size)
        cell_y = int((y - y_min) / (y_max - y_min) * self.grid_size)
        cell_x = max(0, min(self.grid_size - 1, cell_x))
        cell_y = max(0, min(self.grid_size - 1, cell_y))
        
        cell_idx = cell_y * self.grid_size + cell_x
        return self.grid[cell_idx]
    
    def elevation_at_query(self, x: float, y: float) -> Optional[dict]:
        """
        Find elevation at query (x,y) coordinate.
        
        Args:
            x, y: Query coordinates
            
        Returns:
            Dictionary with elevation, slope, aspect, and triangle info, or None if not found
        """
        # Get candidate triangles
        candidate_triangles = self._get_candidate_triangles(x, y)
        
        # Find triangle containing the point
        for face_idx in candidate_triangles:
            face = self.faces[face_idx]
            if len(face) != 3:
                continue
                
            triangle_vertices = [self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]]
            
            if point_in_triangle((x, y), triangle_vertices):
                # Interpolate height using barycentric coordinates
                z = interpolate_height((x, y), triangle_vertices)
                
                return {
                    'elevation': z,
                    'slope': self.triangle_slopes[face_idx],
                    'aspect': self.triangle_aspects[face_idx],
                    'triangle_index': face_idx,
                    'triangle_vertices': triangle_vertices
                }
        
        return None
    
    def nearest_surface_point(self, query_point: np.ndarray) -> dict:
        """
        Find nearest surface point for a 3D query point.
        
        Args:
            query_point: 3D point (x, y, z)
            
        Returns:
            Dictionary with foot point, vertical distance, and surface info
        """
        x, y, z = query_point[0], query_point[1], query_point[2]
        
        # First try to find containing triangle
        elevation_info = self.elevation_at_query(x, y)
        
        if elevation_info is not None:
            # Point projects to inside a triangle
            foot_point = np.array([x, y, elevation_info['elevation']])
            vertical_distance = abs(z - elevation_info['elevation'])
            
            return {
                'foot_point': foot_point,
                'vertical_distance': vertical_distance,
                'surface_elevation': elevation_info['elevation'],
                'slope': elevation_info['slope'],
                'aspect': elevation_info['aspect'],
                'triangle_index': elevation_info['triangle_index'],
                'projection_type': 'inside_triangle'
            }
        
        # Point is outside all triangles, find nearest edge or vertex
        min_distance = float('inf')
        nearest_point = None
        nearest_info = None
        
        # Check all triangles for nearest point on edges
        for face_idx, face in enumerate(self.faces):
            if len(face) != 3:
                continue
                
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            
            # Check each edge
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            for edge_start, edge_end in edges:
                # Project query point to edge
                edge_vec = edge_end - edge_start
                query_vec = np.array([x, y]) - edge_start[:2]  # XY projection only
                
                # Clamp parameter to [0, 1]
                edge_vec_xy = edge_vec[:2]
                dot_product = np.dot(query_vec, edge_vec_xy)
                edge_length_sq = np.dot(edge_vec_xy, edge_vec_xy)
                
                if edge_length_sq > 0:
                    t = dot_product / edge_length_sq
                    t = max(0, min(1, t))
                else:
                    t = 0
                
                # Get point on edge
                edge_point_3d = edge_start + t * edge_vec
                edge_point_xy = edge_point_3d[:2]
                
                # Calculate distance
                distance = np.linalg.norm(np.array([x, y]) - edge_point_xy)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = edge_point_3d
                    nearest_info = {
                        'triangle_index': face_idx,
                        'edge_start': edge_start,
                        'edge_end': edge_end,
                        'parameter': t
                    }
        
        if nearest_point is not None:
            vertical_distance = abs(z - nearest_point[2])
            return {
                'foot_point': nearest_point,
                'vertical_distance': vertical_distance,
                'surface_elevation': nearest_point[2],
                'slope': self.triangle_slopes[nearest_info['triangle_index']],
                'aspect': self.triangle_aspects[nearest_info['triangle_index']],
                'triangle_index': nearest_info['triangle_index'],
                'projection_type': 'nearest_edge',
                'edge_info': nearest_info
            }
        
        # Fallback: return None if no valid surface found
        return None
    
    def shortest_path_hiking(self, start_vertex: int, end_vertex: int, 
                           max_slope_threshold: Optional[float] = None) -> dict:
        """
        Find shortest path with gentle slopes suitable for hiking.
        
        Args:
            start_vertex: Starting vertex index
            end_vertex: Ending vertex index
            max_slope_threshold: Maximum allowed slope in degrees (None for no limit)
            
        Returns:
            Dictionary with path vertices, total distance, and path info
        """
        if start_vertex >= len(self.vertices) or end_vertex >= len(self.vertices):
            return {'error': 'Invalid vertex indices'}
        
        # Build graph with mesh vertices
        graph = self._build_vertex_graph(max_slope_threshold)
        
        # Run Dijkstra's algorithm
        distances, previous = self._dijkstra(graph, start_vertex)
        
        if distances[end_vertex] == float('inf'):
            return {'error': 'No path found between vertices'}
        
        # Reconstruct path
        path = []
        current = end_vertex
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(path) - 1):
            v1, v2 = self.vertices[path[i]], self.vertices[path[i + 1]]
            total_distance += np.linalg.norm(v2 - v1)
        
        return {
            'path_vertices': path,
            'path_coordinates': [self.vertices[v] for v in path],
            'total_distance': total_distance,
            'start_vertex': start_vertex,
            'end_vertex': end_vertex,
            'max_slope_threshold': max_slope_threshold
        }
    
    def _build_vertex_graph(self, max_slope_threshold: Optional[float]) -> dict:
        """Build adjacency graph for vertices with optional slope filtering."""
        graph = {i: [] for i in range(len(self.vertices))}
        
        for face_idx, face in enumerate(self.faces):
            if len(face) != 3:
                continue
                
            # Get face slope
            face_slope = self.triangle_slopes[face_idx]
            
            # Skip face if slope exceeds threshold
            if max_slope_threshold is not None and face_slope > max_slope_threshold:
                continue
            
            # Add edges between vertices of this face
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                if v2 not in graph[v1]:
                    graph[v1].append(v2)
                if v1 not in graph[v2]:
                    graph[v2].append(v1)
        
        return graph
    
    def _dijkstra(self, graph: dict, start: int) -> Tuple[dict, dict]:
        """Dijkstra's algorithm for shortest path."""
        distances = {i: float('inf') for i in range(len(self.vertices))}
        distances[start] = 0
        previous = {i: None for i in range(len(self.vertices))}
        
        unvisited = set(range(len(self.vertices)))
        
        while unvisited:
            # Find unvisited vertex with minimum distance
            current = min(unvisited, key=lambda v: distances[v])
            unvisited.remove(current)
            
            if distances[current] == float('inf'):
                break
            
            # Update distances to neighbors
            for neighbor in graph[current]:
                if neighbor in unvisited:
                    # Calculate 3D edge length
                    edge_length = np.linalg.norm(self.vertices[neighbor] - self.vertices[current])
                    new_distance = distances[current] + edge_length
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        return distances, previous


def load_tin_from_csv(vertices_file: str, faces_file: str, adjacency_file: str) -> GeometricSearchQueries:
    """
    Load TIN data from CSV files and create GeometricSearchQueries instance.
    
    Args:
        vertices_file: Path to vertices CSV file
        faces_file: Path to faces CSV file  
        adjacency_file: Path to adjacency CSV file
        
    Returns:
        GeometricSearchQueries instance
    """
    # Load vertices
    vertices = np.loadtxt(vertices_file, delimiter=',', skiprows=1)
    
    # Load faces
    faces = np.loadtxt(faces_file, delimiter=',', skiprows=1, dtype=int)
    
    # Load adjacency
    adjacency = np.loadtxt(adjacency_file, delimiter=',', skiprows=1, dtype=int)
    
    return GeometricSearchQueries(vertices, faces, adjacency)


def test_geometric_search():
    """Test the geometric search functionality."""
    print("=== Testing Geometric Search Queries ===")
    
    try:
        # Load TIN data
        gsq = load_tin_from_csv('tin_V.csv', 'tin_F.csv', 'tin_A.csv')
        print(f"Loaded TIN with {len(gsq.vertices)} vertices and {len(gsq.faces)} faces")
        
        # Test 1: Elevation at query point
        print("\n1. Testing elevation at query point...")
        test_x, test_y = -0.5, 0.5  # Test point within bounds
        elevation_result = gsq.elevation_at_query(test_x, test_y)
        
        if elevation_result:
            print(f"Point ({test_x}, {test_y}):")
            print(f"  Elevation: {elevation_result['elevation']:.3f}")
            print(f"  Slope: {elevation_result['slope']:.2f}°")
            print(f"  Aspect: {elevation_result['aspect']:.2f}°")
            print(f"  Triangle index: {elevation_result['triangle_index']}")
        else:
            print(f"Point ({test_x}, {test_y}) not found in any triangle")
        
        # Test 2: Nearest surface point
        print("\n2. Testing nearest surface point...")
        test_point = np.array([-0.5, 0.5, 1.0])  # 3D test point within bounds
        nearest_result = gsq.nearest_surface_point(test_point)
        
        if nearest_result:
            print(f"Query point: {test_point}")
            print(f"  Foot point: {nearest_result['foot_point']}")
            print(f"  Vertical distance: {nearest_result['vertical_distance']:.3f}")
            print(f"  Surface elevation: {nearest_result['surface_elevation']:.3f}")
            print(f"  Projection type: {nearest_result['projection_type']}")
        
        # Test 3: Shortest path (if we have enough vertices)
        print("\n3. Testing shortest path...")
        if len(gsq.vertices) >= 2:
            # Find vertices that are definitely connected by looking at faces
            connected_pairs = []
            for face in gsq.faces[:50]:  # Check first 50 faces
                if len(face) == 3:
                    connected_pairs.append((face[0], face[1]))
                    connected_pairs.append((face[1], face[2]))
                    connected_pairs.append((face[2], face[0]))
                    if len(connected_pairs) >= 5:  # Get enough pairs
                        break
            
            # Test without slope filtering first
            if len(connected_pairs) > 0:
                start_vertex, end_vertex = connected_pairs[0]
                path_result = gsq.shortest_path_hiking(start_vertex, end_vertex, max_slope_threshold=None)
                
                if 'error' not in path_result:
                    print(f"Path from vertex {start_vertex} to {end_vertex} (no slope limit):")
                    print(f"  Total distance: {path_result['total_distance']:.3f}")
                    print(f"  Number of vertices: {len(path_result['path_vertices'])}")
                    
                    # Test with slope filtering (using a more reasonable threshold)
                    path_result_filtered = gsq.shortest_path_hiking(start_vertex, end_vertex, max_slope_threshold=80.0)
                    if 'error' not in path_result_filtered:
                        print(f"Path with slope limit 80°:")
                        print(f"  Total distance: {path_result_filtered['total_distance']:.3f}")
                        print(f"  Number of vertices: {len(path_result_filtered['path_vertices'])}")
                    else:
                        print(f"  No path found with slope limit 80°")
                else:
                    print(f"Path error: {path_result['error']}")
            else:
                print("No connected vertex pairs found for testing")
        
        print("\n=== Geometric Search Tests Complete ===")
        
    except FileNotFoundError as e:
        print(f"Error loading TIN files: {e}")
        print("Make sure tin_V.csv, tin_F.csv, and tin_A.csv exist")
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    test_geometric_search()
