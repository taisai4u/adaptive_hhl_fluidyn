# laplacian.py
import numpy as np
from scipy.sparse import diags

def construct_laplacian_2d(grid_size, boundary_conditions='dirichlet'):
    """
    Constructs 2D Laplacian matrix using finite differences (5-point stencil)
    
    Parameters:
    grid_size (tuple): (nx, ny) - number of grid points in x/y directions
    boundary_conditions (str): 'dirichlet' (fixed pressure) or 'neumann' (zero gradient)
    
    Returns:
    scipy.sparse.csr_matrix: Laplacian matrix in compressed sparse row format
    """
    nx, ny = grid_size      #i and j indicies
    n = nx * ny     #n cells
    main_diag = -4 * np.ones(n)
    
    # Create diagonals for finite difference stencil
    diagonals = [
        main_diag,
        np.ones(n-1),        # East/West connections
        np.ones(n-nx),       # North/South connections
    ]
    offsets = [0, 1, nx]
    
    # Handle boundary conditions
    if boundary_conditions == 'dirichlet':
        # Zero out east/west connections (right boundaries)
        for i in range(n - 1):
            if (i + 1) % nx == 0:
                diagonals[1][i] = 0

    # Top boundary (north), just ensure no overwrite beyond array bounds
    # diagonals[2] already doesn't include top row connections, so skip modification

                
    elif boundary_conditions == 'neumann':
        raise NotImplementedError("Neumann BCs require ghost nodes - start with Dirichlet")
    
    # Build sparse matrix
    laplacian = diags(diagonals, offsets, shape=(n, n), format='csr')
    return laplacian

def analyze_matrix(matrix):
    """Basic matrix analysis for HHL suitability"""
    from scipy.sparse.linalg import eigsh
    analysis = {
        'condition_number': None,
        'min_eigenvalue': None,
        'max_eigenvalue': None,
        'sparsity': 1 - (matrix.nnz / (matrix.shape[0]*matrix.shape[1]))
    }
    
    # Compute extreme eigenvalues
    eigenvalues = eigsh(matrix, k=2, which='BE', return_eigenvectors=False)
    analysis['min_eigenvalue'] = eigenvalues[0]
    analysis['max_eigenvalue'] = eigenvalues[1]
    analysis['condition_number'] = eigenvalues[1]/eigenvalues[0]
    
    return analysis

if __name__ == "__main__":
    # Simple test case: 3x3 grid with Dirichlet BCs
    laplacian = construct_laplacian_2d((3, 3))
    print("Laplacian matrix (dense view):\n", laplacian.toarray())
    print("\nMatrix analysis:", analyze_matrix(laplacian))