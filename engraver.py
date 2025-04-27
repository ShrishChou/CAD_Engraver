import numpy as np
from stl import mesh
import math
# import freetype
import cv2
def create_extruded_array(array, square_size=7.5, height=2.0):
    """
    Creates an STL file from an 8x8 binary array where 1s are represented as extruded squares.
    
    Parameters:
    array: 8x8 numpy array of 0s and 1s
    square_size: size of each square in mm (default 7.5)
    height: height of extrusion in mm (default 2.0)
    """
    # Initialize lists to store vertices and faces
    vertices = []
    faces = []
    vertex_count = 0

    # Create base plate (optional - comment out if not needed)
    # base_size = 8 * square_size
    # base_vertices = [
    #     [0, 0, 0],
    #     [base_size, 0, 0],
    #     [base_size, base_size, 0],
    #     [0, base_size, 0]
    # ]
    # vertices.extend(base_vertices)
    # faces.extend([
    #     [0, 1, 2],  # First triangle
    #     [0, 2, 3]   # Second triangle
    # ])
    # vertex_count += 4

    # Process each cell in the array
    for i in range(8):
        for j in range(8):
            if array[i][j] == 1:
                # Calculate position of current square
                x = j * square_size
                y = i * square_size
                z = 0

                # Create vertices for extruded square
                new_vertices = [
                    # Bottom vertices
                    [x, y, z],
                    [x + square_size, y, z],
                    [x + square_size, y + square_size, z],
                    [x, y + square_size, z],
                    # Top vertices
                    [x, y, z + height],
                    [x + square_size, y, z + height],
                    [x + square_size, y + square_size, z + height],
                    [x, y + square_size, z + height]
                ]
                vertices.extend(new_vertices)

                # Create faces (12 triangles for each cube)
                cube_faces = [
                    # Bottom face
                    [vertex_count + 0, vertex_count + 1, vertex_count + 2],
                    [vertex_count + 0, vertex_count + 2, vertex_count + 3],
                    # Top face
                    [vertex_count + 4, vertex_count + 5, vertex_count + 6],
                    [vertex_count + 4, vertex_count + 6, vertex_count + 7],
                    # Side faces
                    [vertex_count + 0, vertex_count + 1, vertex_count + 5],
                    [vertex_count + 0, vertex_count + 5, vertex_count + 4],
                    [vertex_count + 1, vertex_count + 2, vertex_count + 6],
                    [vertex_count + 1, vertex_count + 6, vertex_count + 5],
                    [vertex_count + 2, vertex_count + 3, vertex_count + 7],
                    [vertex_count + 2, vertex_count + 7, vertex_count + 6],
                    [vertex_count + 3, vertex_count + 0, vertex_count + 4],
                    [vertex_count + 3, vertex_count + 4, vertex_count + 7]
                ]
                faces.extend(cube_faces)
                vertex_count += 8

    # Convert to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create the mesh
    cube = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]

    return cube

# Example usage:
def combine_with_base(array, base_stl_path="housingchanged.stl", output_file="output.stl", 
                      pattern_position=(0, 0, 0), rotation=(0, 0, 0)):
    """
    Generate pattern and combine it with an existing base STL file.
    
    Parameters:
    array: 8x8 numpy array of 0s and 1s
    base_stl_path: path to the base STL file
    output_file: name of the output combined STL file
    pattern_position: (x, y, z) position to place the pattern on the base
    rotation: (rx, ry, rz) rotation in degrees to apply to the pattern
    """
    from stl import mesh
    import numpy as np
    from math import radians, cos, sin
    
    # Load the base mesh
    base_mesh = mesh.Mesh.from_file(base_stl_path)
    
    # Create the pattern mesh
    pattern_mesh = create_extruded_array(array)
    
    # Apply rotation to pattern if needed
    if any(rotation):
        # Convert degrees to radians
        rx, ry, rz = map(radians, rotation)
        
        # Rotation matrices
        def rotation_matrix(rx, ry, rz):
            Rx = np.array([[1, 0, 0],
                          [0, cos(rx), -sin(rx)],
                          [0, sin(rx), cos(rx)]])
            
            Ry = np.array([[cos(ry), 0, sin(ry)],
                          [0, 1, 0],
                          [-sin(ry), 0, cos(ry)]])
            
            Rz = np.array([[cos(rz), -sin(rz), 0],
                          [sin(rz), cos(rz), 0],
                          [0, 0, 1]])
            
            return Rz.dot(Ry.dot(Rx))
        
        R = rotation_matrix(rx, ry, rz)
        pattern_mesh.vectors = pattern_mesh.vectors.dot(R)
    
    # Apply translation to pattern
    x, y, z = pattern_position
    pattern_mesh.vectors += np.array([x, y, z])
    
    # Combine meshes
    combined = mesh.Mesh(np.concatenate([base_mesh.data, pattern_mesh.data]))
    
    # Save combined mesh
    combined.save(output_file)
    
def generate_stl_file(array, filename="output.stl"):
    """
    Generate an STL file from an 8x8 binary array.
    
    Parameters:
    array: 8x8 numpy array of 0s and 1s
    filename: output STL filename
    """
    # Convert input to numpy array if it isn't already
    array = np.array(array)
    
    # Verify array dimensions
    if array.shape != (8, 8):
        raise ValueError("Array must be 8x8")
    
    # Create the mesh
    cube = create_extruded_array(array)
    
    # Save the mesh
    cube.save(filename)
def create_line_segment(start, end, width, height, depth):
    """Create a rectangular prism for a line segment of a digit"""
    # Calculate the direction and length
    direction = end - start
    length = np.linalg.norm(direction)
    direction = direction / length
    
    # Calculate perpendicular vector for width
    perpendicular = np.array([-direction[1], direction[0]])
    
    # Create vertices
    vertices = np.array([
        start + perpendicular * width/2,
        start - perpendicular * width/2,
        end + perpendicular * width/2,
        end - perpendicular * width/2,
    ])
    
    # Create 3D vertices (bottom and top)
    vertices_3d = []
    for v in vertices:
        vertices_3d.extend([
            [v[0], v[1], 0],
            [v[0], v[1], height]
        ])
    
    vertices = np.array(vertices_3d)
    
    # Create faces
    faces = np.array([
        [0, 1, 2], [1, 3, 2],  # Front
        [4, 6, 5], [5, 6, 7],  # Back
        [0, 2, 4], [2, 6, 4],  # Right
        [1, 5, 3], [3, 5, 7],  # Left
        [2, 3, 6], [3, 7, 6],  # Top
        [0, 4, 1], [1, 4, 5]   # Bottom
    ])
    
    return vertices, faces

def create_digit(digit, size=5.0, line_width=1.0, height=1.0):
    """Create a single digit with the given size using improved geometry"""
    # Define points for better digit formation
    points = {
        0: np.array([0.2, 1.0]),     # Top left
        1: np.array([0.8, 1.0]),     # Top right
        2: np.array([0.8, 0.55]),    # Upper middle right
        3: np.array([0.8, 0.0]),     # Bottom right
        4: np.array([0.2, 0.0]),     # Bottom left
        5: np.array([0.2, 0.55]),    # Upper middle left
        6: np.array([0.2, 0.45]),    # Lower middle left
        7: np.array([0.8, 0.45]),    # Lower middle right
        8: np.array([0.5, 0.5]),     # Center
        9: np.array([0.5, 1.0]),     # Top middle
    }
    
    # Define segments for each digit with improved paths
    segments = {
        '0': [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)],         # More rectangular 0
        '1': [(9,3)],                                             # Straight 1
        '2': [(0,1), (1,2), (2,8), (8,6), (6,4), (4,3)],        # Curved 2 with proper sweep
        '3': [(0,1), (1,2), (2,8), (8,7), (7,3), (3,4)],        # Proper 3 with middle indent
        '4': [(5,2), (2,7), (0,7), (7,3)],                       # Clear 4 with proper cross
        '5': [(1,0), (0,5), (5,8), (8,7), (7,3), (3,4)],        # Better curved 5
        '6': [(1,0), (0,4), (4,3), (3,7), (7,8), (8,6)],        # Proper 6 with loop
        '7': [(0,1), (1,3)],                                      # Simple clean 7
        '8': [(0,1), (1,2), (2,7), (7,3), (3,4), (4,6), (6,5), (5,0)], # Full 8 with middle pinch
        '9': [(4,3), (3,7), (7,2), (2,1), (1,0), (0,5)]         # Proper 9 with loop
    }
    
    # Scale points
    scaled_points = {k: v * size for k, v in points.items()}
    
    # Create segments for the digit
    all_vertices = []
    all_faces = []
    vertex_count = 0
    
    for start_idx, end_idx in segments[str(digit)]:
        start = scaled_points[start_idx]
        end = scaled_points[end_idx]
        
        vertices, faces = create_line_segment(start, end, line_width, height, 0)
        
        # Adjust faces for current vertex count
        faces = faces + vertex_count
        
        all_vertices.extend(vertices)
        all_faces.extend(faces)
        vertex_count += len(vertices)
    
    return np.array(all_vertices), np.array(all_faces)

def create_number_mesh(number, digit_size=5.0, spacing=1.0, height=1.0, line_width=1.0):
    """Create a mesh for a multi-digit number"""
    number_str = str(number)
    all_vertices = []
    all_faces = []
    vertex_count = 0
    
    for i, digit in enumerate(number_str):
        vertices, faces = create_digit(digit, digit_size, line_width, height)
        
        # Offset vertices for current digit position
        vertices = vertices + np.array([i * (digit_size + spacing), 0, 0])
        
        # Adjust faces for current vertex count
        faces = faces + vertex_count
        
        all_vertices.extend(vertices)
        all_faces.extend(faces)
        vertex_count += len(vertices)
    
    # Create the mesh
    number_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    vertices = np.array(all_vertices)
    
    for i, f in enumerate(all_faces):
        for j in range(3):
            number_mesh.vectors[i][j] = vertices[f[j],:]
    
    return number_mesh

def add_number_to_stl(base_stl_path, number, output_path, 
                     position=(0, 0, 0), rotation=(0, 0, 0),
                     digit_size=5.0, spacing=1.0, height=1.0, line_width=1.0):
    """Add extruded number to an existing STL file"""
    # Load base mesh
    base_mesh = mesh.Mesh.from_file(base_stl_path)
    
    # Create number mesh
    number_mesh = create_number_mesh(number, digit_size, spacing, height, line_width)
    
    # Apply rotation if needed
    if any(rotation):
        # Convert degrees to radians
        rx, ry, rz = map(np.radians, rotation)
        
        # Create rotation matrices
        def rotation_matrix(rx, ry, rz):
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(rx), -np.sin(rx)],
                          [0, np.sin(rx), np.cos(rx)]])
            
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                          [0, 1, 0],
                          [-np.sin(ry), 0, np.cos(ry)]])
            
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                          [np.sin(rz), np.cos(rz), 0],
                          [0, 0, 1]])
            
            return Rz.dot(Ry.dot(Rx))
        
        R = rotation_matrix(rx, ry, rz)
        number_mesh.vectors = number_mesh.vectors.dot(R)
    
    # Apply translation
    number_mesh.vectors += np.array(position)
    
    # Combine meshes
    combined = mesh.Mesh(np.concatenate([base_mesh.data, number_mesh.data]))
    
    # Save result
    combined.save(output_path)

def reflect_matrix_horizontally(matrix):
    """
    Reflects an 8x8 matrix around its horizontal middle line.
    
    Parameters:
    matrix: 8x8 numpy array or list of lists
    
    Returns:
    reflected_matrix: 8x8 numpy array with horizontal reflection
    """
    # Convert to numpy array if it isn't already
    matrix = np.array(matrix)
    
    # Verify the matrix is 8x8
    if matrix.shape != (8, 8):
        raise ValueError("Matrix must be 8x8")
    
    # Create the reflected matrix by flipping rows
    reflected_matrix = np.flipud(matrix)
    
    return reflected_matrix


def generate_aruco_pattern(aruco_id):
    """
    Generate an Aruco pattern matrix.
    
    Parameters:
    aruco_id (int): ID of the Aruco marker to generate.
    size (int): Size of the Aruco marker (default 6x6).
    
    Returns:
    numpy.ndarray: Binary matrix representing the Aruco pattern.
    """
    # Define the dictionary

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Create an image from the marker
    tag_size = 400  # Size of the tag in pixels
    tag_image = np.zeros((tag_size, tag_size, 1), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, aruco_id, tag_size, tag_image, 1)

    # Save the image
    # cv2.imwrite('aruco_tag_'+str(aruco_id)+'.png', tag_image)

    # Optional: Display the image
    # cv2.imshow("ArUco Tag 1s", tag_image)
    # cv2.waitKey(0)
    
    marker_grid_size = 8  # Number of cells in one dimension for DICT_6X6_250
    binary_pattern = cv2.resize(tag_image, (marker_grid_size, marker_grid_size), interpolation=cv2.INTER_NEAREST)

    # Threshold the resized image to create a binary pattern
    binary_pattern = (binary_pattern > 127).astype(np.uint8)

    # Print the binary pattern
    # print("Binary Pattern:\n", binary_pattern)

    # Optional: Display the binary pattern
    # cv2.imshow("Binary Pattern", cv2.resize(binary_pattern * 255, (200, 200), interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    binary_pattern=reflect_matrix_horizontally(binary_pattern)
    return binary_pattern

   

tag_id=int(input("Input Aruco Tag: "))
pattern=generate_aruco_pattern(tag_id)
# Generate the STL file
combine_with_base(
    array=pattern,
    base_stl_path="new housing 1-29.stl",
    pattern_position=(-30, -30, 102),
    rotation=(0, 0, 0),
    output_file=f'Printing_Tag{tag_id}.stl'
)
print(f"Find data at Printing_Tag{tag_id}.stl")
# combine_text_with_base(
#         text="1234",
#         base_stl_path="combined.stl",
#         output_path="trial_with_extruded_text.stl",
#         position=(-10, 0, 12),     # Adjust position as needed
#         rotation=(0, 0, 0),       # Adjust rotation as needed
#         text_height=1,          # Text height in mm
#         extrusion_height=1.0      # How far the text protrudes in mm
# )
# add_number_to_stl(
#         base_stl_path="combined.stl",
#         number=1234,
#         output_path="trial_with_number.stl",
#         position=(10, 10, 1),    # Position coordinates
#         rotation=(0, 0, 0),      # Rotation angles
#         digit_size=2,          # Size of each digit (mm)
#         spacing=1.5,             # Space between digits (mm)
#         height=2,              # Extrusion height (mm)
#         line_width=0.5           # Width of the lines making up the digits (mm)
#     )