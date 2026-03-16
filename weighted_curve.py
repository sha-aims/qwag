from sage.all import gcd, PolynomialRing
from itertools import product
from sage.all import LinearCode # Import SageMath's LinearCode

from .weighted_projective_space import WeightedProjectiveSpace  # Corrected import

# Define our custom LinearCode class
class CustomLinearCode:
    def __init__(self, generator_matrix):
        self.generator_matrix = generator_matrix
        self.n = generator_matrix.ncols()
        self.k = generator_matrix.rank() # Correctly calculate k as the rank of the generator matrix

    def __repr__(self):
        return f"CustomLinearCode with parameters [n={self.n}, k={self.k}]\nGenerator matrix:\n{self.generator_matrix}"

    def is_self_orthogonal(self):
        """
        Checks if the linear code is self-orthogonal, i.e., C subset C^perp.
        This means that every codeword is orthogonal to every other codeword (including itself).
        For a generator matrix G, this is equivalent to G * G^T == 0.
        """
        if self.generator_matrix.is_zero():
            return True

        result_matrix = self.generator_matrix * self.generator_matrix.transpose()
        return result_matrix.is_zero()

    def dual_code(self):
        """
        Returns the dual code (C^perp) of the current linear code.
        The dual code is defined as the set of all vectors that are orthogonal to every codeword in the original code.
        """
        # Create a SageMath LinearCode instance using the globally imported Sage LinearCode
        sage_code = LinearCode(self.generator_matrix)

        # Compute the dual code using SageMath's functionality
        sage_dual_code = sage_code.dual_code()

        # Return a new instance of our custom CustomLinearCode class with the dual code's generator matrix
        return CustomLinearCode(sage_dual_code.generator_matrix())

# Helper function for generating weighted monomials (free function in the module)
def _generate_weighted_monomials(poly_ring, weights, max_weighted_degree):
    """
    Generates a list of monomials in `poly_ring` whose weighted degree is less than or equal to
    `max_weighted_degree`.
    """
    monomials = []
    num_vars = len(weights)
    variables = list(poly_ring.gens())

    if max_weighted_degree < 0:
        return []

    def __generate_recursive_exponents(current_exponents_list, current_weighted_degree, var_idx):
        if var_idx == num_vars:
            monomial = poly_ring(1)
            for i in range(num_vars):
                monomial *= variables[i]**current_exponents_list[i]
            monomials.append(monomial)
            return

        if weights[var_idx] == 0:
            max_exponent_for_var = 0
        else:
            max_exponent_for_var = (max_weighted_degree - current_weighted_degree) // weights[var_idx]

        for exponent in range(max_exponent_for_var + 1):
            __generate_recursive_exponents(
                current_exponents_list + [exponent],
                current_weighted_degree + exponent * weights[var_idx],
                var_idx + 1
            )

    __generate_recursive_exponents([], 0, 0)
    return monomials


class WeightedCurve(WeightedProjectiveSpace):
    def __init__(self, space, polynomial):
        """
        Weighted curve defined by a weighted homogeneous polynomial in the space.
        space: WeightedProjectiveSpace instance
        polynomial: weighted homogeneous polynomial (Sage poly)
        """
        super().__init__(space.field, space.weights)
        self.space = space # Store the space object
        self.poly = polynomial

        # Calculate weighted degree and check for weighted homogeneity
        weighted_degrees_of_terms = []
        if not self.poly: # Handle zero polynomial
            raise ValueError("Polynomial cannot be zero.")

        for exponents_tuple, coeff in self.poly.dict().items():
            term_weighted_degree = sum(exponent * weight for exponent, weight in zip(exponents_tuple, self.weights))
            weighted_degrees_of_terms.append(term_weighted_degree)

        if not weighted_degrees_of_terms:
            raise ValueError("Polynomial is empty (should not happen for non-zero poly).")

        deg = weighted_degrees_of_terms[0]
        if not all(d == deg for d in weighted_degrees_of_terms):
            raise ValueError("Polynomial is not weighted homogeneous with given weights.")

        if deg <= 0:
            raise ValueError("Polynomial must have positive weighted degree.")

        self.weighted_degree = deg # Store the weighted degree

        # Coordinate ring is inherited from space
        self.coord_ring = space.coord_ring

    def __repr__(self):
        return f"WeightedCurve in {self.space} defined by {self.poly}"

    def arithmetic_genus(self):
        """
        Computes the arithmetic genus of the curve.
        This formula is applicable for a smooth hypersurface of weighted degree `d`
        in a weighted projective space of dimension `len(self.weights) - 1`.
        Note: This implementation does not account for singularities and provides the arithmetic genus.
        """
        d = self.weighted_degree
        ambient_dim = len(self.weights) - 1 # Dimension of the ambient weighted projective space
        g = 1 + (d - 2) * ambient_dim // 2
        return g

    def smooth_rational_points(self):
        """
        Returns smooth (non-singular) rational points on the curve.
        This implementation iterates over affine points and checks for vanishing
        of the polynomial and its partial derivatives.
        Note: This currently returns affine points. Proper projective normalization
        for weighted projective space is more complex and not implemented here.
        """
        field = self.field
        coord_ring = self.coord_ring
        variables = list(coord_ring.gens())
        num_vars = len(self.weights) # n from (x_0, ..., x_{n-1})

        smooth_points = []

        # Calculate partial derivatives
        partial_derivatives = [self.poly.derivative(var) for var in variables]

        # Iterate over all possible affine points (x_0, ..., x_{num_vars-1})
        for p_coords_tuple in product(field, repeat=num_vars):
            # Create a substitution dictionary
            sub_dict = {var: val for var, val in zip(variables, p_coords_tuple)}

            # 1. Check if the point is on the curve (F(P) = 0)
            if self.poly.subs(sub_dict) == 0:
                # 2. Check for smoothness (not all partial derivatives vanish at P)
                if not self.is_singular(p_coords_tuple):
                    smooth_points.append(p_coords_tuple)

        return smooth_points

    def is_singular(self, p):
        """
        Checks if a given point p (tuple of coordinates) on the curve is singular.
        A point is singular if all partial derivatives of the polynomial vanish at that point.
        """
        coord_ring = self.coord_ring
        variables = list(coord_ring.gens())

        # Create a substitution dictionary for the point p
        sub_dict = {var: val for var, val in zip(variables, p)}

        # Calculate partial derivatives (if not already cached)
        partial_derivatives = [self.poly.derivative(var) for var in variables]

        # Check if all partial derivatives vanish at point p
        for derivative_poly in partial_derivatives:
            if derivative_poly.subs(sub_dict) != 0:
                return False  # Not singular, at least one derivative is non-zero
        return True  # All partial derivatives vanish, so it's singular

    def rational_points(self):
        """
        Returns a list of all distinct rational points on the curve.
        It retrieves all rational points from the ambient weighted projective space
        and filters them to include only those that satisfy the curve's equation.
        """
        curve_points = []
        ambient_points = self.space.rational_points() # Get points from the ambient space

        # Get variables from the coordinate ring to create substitution dictionaries
        variables = list(self.coord_ring.gens())

        for p_coords_tuple in ambient_points:
            # Create a substitution dictionary for the point p
            sub_dict = {var: val for var, val in zip(variables, p_coords_tuple)}

            # Check if the point satisfies the curve's polynomial (F(P) = 0)
            if self.poly.subs(sub_dict) == 0:
                curve_points.append(p_coords_tuple)

        return sorted(curve_points)

    def singular_rational_points(self):
        """
        Returns a list of singular rational points on the curve.
        These are points on the curve where all partial derivatives of the defining
        polynomial vanish simultaneously.
        """
        singular_points = []
        all_rational_points = self.rational_points() # Get all points on the curve

        for p_coords_tuple in all_rational_points:
            if self.is_singular(p_coords_tuple):
                singular_points.append(p_coords_tuple)
        return sorted(singular_points)

    def geometric_genus(self):
        """
        Approximates the geometric genus (p_g) of the curve.
        For a curve with ordinary singularities, the geometric genus can be estimated
        as p_g = p_a - sum(delta_i) where p_a is the arithmetic genus and delta_i
        are contributions from each singularity. For this approximation, we will
        simply subtract the number of singular rational points from the arithmetic genus.

        Disclaimer: This is a highly simplified approximation and does not account
        for the complex nature and multiplicity of singularities. It serves as a
        basic lower bound for the geometric genus for non-singular cases, and
        a rough estimation for singular cases. For precise geometric genus, more
        advanced techniques are required.
        """
        # A very simplified approximation: geometric genus <= arithmetic genus
        # and roughly p_g = p_a - number_of_singular_points
        # This assumes each singular point reduces the genus by 1, which is not generally true
        # for all types of singularities or for non-ordinary singularities.
        num_singular_points = len(self.singular_rational_points())
        approx_geometric_genus = self.arithmetic_genus() - num_singular_points
        return max(0, approx_geometric_genus) # Genus cannot be negative

    def riemann_roch_basis(self, divisor_degree):
        """
        Computes a basis for the Riemann-Roch space L(D) for a divisor D of degree `divisor_degree`.
        """
        if divisor_degree < 0:
            return []
        if divisor_degree == 0:
            return [self.coord_ring(1)] # Space of constant functions

        # Generate candidate monomials up to the desired weighted degree
        candidate_monomials = _generate_weighted_monomials(
            self.coord_ring, self.weights, divisor_degree
        )

        # Compute a Groebner basis for the ideal generated by the curve polynomial
        ideal = self.coord_ring.ideal(self.poly)
        gb = ideal.groebner_basis()

        # Reduce all candidate monomials modulo the Groebner basis to find unique representatives
        basis_elements_reduced = set()
        for monomial in candidate_monomials:
            reduced_form = monomial.reduce(gb)
            basis_elements_reduced.add(reduced_form)

        # Sort the unique basis elements for consistent output
        return sorted(list(basis_elements_reduced), key=str)

    def evaluation_code(self, points_list, divisor_degree):
        """
        Creates a generator matrix for a linear code based on evaluating
        functions from a Riemann-Roch space at a set of rational points.
        """
        from sage.all import matrix # Import matrix from sage.all

        basis = self.riemann_roch_basis(divisor_degree)
        gen_matrix = []

        if not basis or not points_list:
            return CustomLinearCode(matrix(self.field, 0, 0)) # Return empty CustomLinearCode

        for f in basis:
            row = []
            for p in points_list:
                variables = list(self.coord_ring.gens())
                sub_dict = {var: val for var, val in zip(variables, p)}
                row.append(f.subs(sub_dict))
            gen_matrix.append(row)

        if not gen_matrix or not gen_matrix[0]:
            return CustomLinearCode(matrix(self.field, 0, 0))

        G_matrix = matrix(self.field, gen_matrix)

        # Return a CustomLinearCode object
        return CustomLinearCode(G_matrix.echelon_form())

# Function to construct CSS quantum code
def construct_css_quantum_code(classical_code):
    """
    Constructs a CSS (Calderbank-Shor-Steane) quantum code from a self-orthogonal classical linear code.
    """
    assert classical_code.is_self_orthogonal(), "The classical code must be self-orthogonal for CSS construction."

    C_Z = classical_code
    C_X = classical_code.dual_code()

    n_Q = C_Z.n
    k_Q = C_X.k - C_Z.k
    d_Q = None

    return {
        'n_Q': n_Q,
        'k_Q': k_Q,
        'd_Q': d_Q,
        'C_X': C_X,
        'C_Z': C_Z
    }
