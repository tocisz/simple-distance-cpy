"""
=====================================================
Distance computations (:mod:`scipy.spatial.distance`)
=====================================================

.. sectionauthor:: Damian Eads

Function Reference
------------------

Distance matrix computation from a collection of raw observation vectors
stored in a rectangular array.

.. autosummary::
   :toctree: generated/

   pdist   -- pairwise distances between observation vectors.
   cdist   -- distances between two collections of observation vectors
   squareform -- convert distance matrix to a condensed one and vice versa

Predicates for checking the validity of distance matrices, both
condensed and redundant. Also contained in this module are functions
for computing the number of observations in a distance matrix.

.. autosummary::
   :toctree: generated/

   is_valid_dm -- checks for a valid distance matrix
   is_valid_y  -- checks for a valid condensed distance matrix
   num_obs_dm  -- # of observations in a distance matrix
   num_obs_y   -- # of observations in a condensed distance matrix

Distance functions between two numeric vectors ``u`` and ``v``. Computing
distances over a large collection of vectors is inefficient for these
functions. Use ``pdist`` for this purpose.

.. autosummary::
   :toctree: generated/

   braycurtis       -- the Bray-Curtis distance.
   canberra         -- the Canberra distance.
   chebyshev        -- the Chebyshev distance.
   cityblock        -- the Manhattan distance.
   correlation      -- the Correlation distance.
   cosine           -- the Cosine distance.
   euclidean        -- the Euclidean distance.
   mahalanobis      -- the Mahalanobis distance.
   minkowski        -- the Minkowski distance.
   seuclidean       -- the normalized Euclidean distance.
   sqeuclidean      -- the squared Euclidean distance.
   wminkowski       -- the weighted Minkowski distance.

Distance functions between two boolean vectors (representing sets) ``u`` and
``v``.  As in the case of numerical vectors, ``pdist`` is more efficient for
computing the distances between all pairs.

.. autosummary::
   :toctree: generated/

   dice             -- the Dice dissimilarity.
   hamming          -- the Hamming distance.
   jaccard          -- the Jaccard distance.
   kulsinski        -- the Kulsinski distance.
   matching         -- the matching dissimilarity.
   rogerstanimoto   -- the Rogers-Tanimoto dissimilarity.
   russellrao       -- the Russell-Rao dissimilarity.
   sokalmichener    -- the Sokal-Michener dissimilarity.
   sokalsneath      -- the Sokal-Sneath dissimilarity.
   yule             -- the Yule dissimilarity.

:func:`hamming` also operates over discrete numerical vectors.
"""

# Copyright (C) Damian Eads, 2007-2008. New BSD License.

from __future__ import division, print_function, absolute_import

__all__ = [
    'cdist'
]

import warnings
import numpy as np

import _distance_wrap

def _convert_to_double(X):
    if X.dtype != np.double:
        X = X.astype(np.double)
    if not X.flags.contiguous:
        X = X.copy()
    return X

# Registry of "simple" distance metrics' pdist and cdist implementations,
# meaning the ones that accept one dtype and have no additional arguments.
_SIMPLE_CDIST = {}
_SIMPLE_PDIST = {}

for names, wrap_name in [
    (['braycurtis'], "bray_curtis"),
    (['canberra'], "canberra"),
    (['chebychev', 'chebyshev', 'cheby', 'cheb', 'ch'], "chebyshev"),
    (["cityblock", "cblock", "cb", "c"], "city_block"),
    (["euclidean", "euclid", "eu", "e"], "euclidean"),
    (["sqeuclidean", "sqe", "sqeuclid"], "sqeuclidean"),
]:
    cdist_fn = getattr(_distance_wrap, "cdist_%s_wrap" % wrap_name)
    pdist_fn = getattr(_distance_wrap, "pdist_%s_wrap" % wrap_name)
    for name in names:
        _SIMPLE_CDIST[name] = _convert_to_double, cdist_fn
        _SIMPLE_PDIST[name] = _convert_to_double, pdist_fn

for name in ["dice", "kulsinski", "matching", "rogerstanimoto", "russellrao",
             "sokalmichener", "sokalsneath", "yule"]:
    wrap_name = "hamming" if name == "matching" else name

    cdist_fn = getattr(_distance_wrap, "cdist_%s_bool_wrap" % wrap_name)
    _SIMPLE_CDIST[name] = _convert_to_bool, cdist_fn

    pdist_fn = getattr(_distance_wrap, "pdist_%s_bool_wrap" % wrap_name)
    _SIMPLE_PDIST[name] = _convert_to_bool, pdist_fn


def cdist(XA, XB, metric='euclidean', p=2, V=None, VI=None, w=None):
    """
    Computes distance between each pair of the two collections of inputs.

    The following are common calling conventions:

    1. ``Y = cdist(XA, XB, 'euclidean')``

       Computes the distance between :math:`m` points using
       Euclidean distance (2-norm) as the distance metric between the
       points. The points are arranged as :math:`m`
       :math:`n`-dimensional row vectors in the matrix X.

    2. ``Y = cdist(XA, XB, 'minkowski', p)``

       Computes the distances using the Minkowski distance
       :math:`||u-v||_p` (:math:`p`-norm) where :math:`p \\geq 1`.

    3. ``Y = cdist(XA, XB, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}.

       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points. If not passed, it is
       automatically computed.

    5. ``Y = cdist(XA, XB, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`||u-v||_2^2` between
       the vectors.

    6. ``Y = cdist(XA, XB, 'cosine')``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \\frac{u \\cdot v}
                   {{||u||}_2 {||v||}_2}

       where :math:`||*||_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of :math:`u` and :math:`v`.

    7. ``Y = cdist(XA, XB, 'correlation')``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}

       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.


    8. ``Y = cdist(XA, XB, 'hamming')``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = cdist(XA, XB, 'jaccard')``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree where at least one of them is non-zero.

    10. ``Y = cdist(XA, XB, 'chebyshev')``

       Computes the Chebyshev distance between the points. The
       Chebyshev distance between two n-vectors ``u`` and ``v`` is the
       maximum norm-1 distance between their respective elements. More
       precisely, the distance is given by

       .. math::

          d(u,v) = \\max_i {|u_i-v_i|}.

    11. ``Y = cdist(XA, XB, 'canberra')``

       Computes the Canberra distance between the points. The
       Canberra distance between two points ``u`` and ``v`` is

       .. math::

         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.

    12. ``Y = cdist(XA, XB, 'braycurtis')``

       Computes the Bray-Curtis distance between the points. The
       Bray-Curtis distance between two points ``u`` and ``v`` is


       .. math::

            d(u,v) = \\frac{\\sum_i (u_i-v_i)}
                          {\\sum_i (u_i+v_i)}

    13. ``Y = cdist(XA, XB, 'mahalanobis', VI=None)``

       Computes the Mahalanobis distance between the points. The
       Mahalanobis distance between two points ``u`` and ``v`` is
       :math:`(u-v)(1/V)(u-v)^T` where :math:`(1/V)` (the ``VI``
       variable) is the inverse covariance. If ``VI`` is not None,
       ``VI`` will be used as the inverse covariance matrix.

    14. ``Y = cdist(XA, XB, 'yule')``

       Computes the Yule distance between the boolean
       vectors. (see `yule` function documentation)

    15. ``Y = cdist(XA, XB, 'matching')``

       Synonym for 'hamming'.

    16. ``Y = cdist(XA, XB, 'dice')``

       Computes the Dice distance between the boolean vectors. (see
       `dice` function documentation)

    17. ``Y = cdist(XA, XB, 'kulsinski')``

       Computes the Kulsinski distance between the boolean
       vectors. (see `kulsinski` function documentation)

    18. ``Y = cdist(XA, XB, 'rogerstanimoto')``

       Computes the Rogers-Tanimoto distance between the boolean
       vectors. (see `rogerstanimoto` function documentation)

    19. ``Y = cdist(XA, XB, 'russellrao')``

       Computes the Russell-Rao distance between the boolean
       vectors. (see `russellrao` function documentation)

    20. ``Y = cdist(XA, XB, 'sokalmichener')``

       Computes the Sokal-Michener distance between the boolean
       vectors. (see `sokalmichener` function documentation)

    21. ``Y = cdist(XA, XB, 'sokalsneath')``

       Computes the Sokal-Sneath distance between the vectors. (see
       `sokalsneath` function documentation)


    22. ``Y = cdist(XA, XB, 'wminkowski')``

       Computes the weighted Minkowski distance between the
       vectors. (see `wminkowski` function documentation)

    23. ``Y = cdist(XA, XB, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         dm = cdist(XA, XB, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function `sokalsneath`. This would result in
       sokalsneath being called :math:`{n \\choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax::

         dm = cdist(XA, XB, 'sokalsneath')

    Parameters
    ----------
    XA : ndarray
        An :math:`m_A` by :math:`n` array of :math:`m_A`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    XB : ndarray
        An :math:`m_B` by :math:`n` array of :math:`m_B`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
    w : ndarray, optional
        The weight vector (for weighted Minkowski).
    p : scalar, optional
        The p-norm to apply (for Minkowski, weighted and unweighted)
    V : ndarray, optional
        The variance vector (for standardized Euclidean).
    VI : ndarray, optional
        The inverse of the covariance matrix (for Mahalanobis).

    Returns
    -------
    Y : ndarray
        A :math:`m_A` by :math:`m_B` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.

    Raises
    ------
    ValueError
        An exception is thrown if `XA` and `XB` do not have
        the same number of columns.

    Examples
    --------
    Find the Euclidean distances between four 2-D coordinates:

    >>> from scipy.spatial import distance
    >>> coords = [(35.0456, -85.2672),
    ...           (35.1174, -89.9711),
    ...           (35.9728, -83.9422),
    ...           (36.1667, -86.7833)]
    >>> distance.cdist(coords, coords, 'euclidean')
    array([[ 0.    ,  4.7044,  1.6172,  1.8856],
           [ 4.7044,  0.    ,  6.0893,  3.3561],
           [ 1.6172,  6.0893,  0.    ,  2.8477],
           [ 1.8856,  3.3561,  2.8477,  0.    ]])


    Find the Manhattan distance from a 3-D point to the corners of the unit
    cube:

    >>> a = np.array([[0, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 1, 0],
    ...               [0, 1, 1],
    ...               [1, 0, 0],
    ...               [1, 0, 1],
    ...               [1, 1, 0],
    ...               [1, 1, 1]])
    >>> b = np.array([[ 0.1,  0.2,  0.4]])
    >>> distance.cdist(a, b, 'cityblock')
    array([[ 0.7],
           [ 0.9],
           [ 1.3],
           [ 1.5],
           [ 1.5],
           [ 1.7],
           [ 2.1],
           [ 2.3]])

    """
    # You can also call this as:
    #     Y = cdist(XA, XB, 'test_abc')
    # where 'abc' is the metric being tested.  This computes the distance
    # between all pairs of vectors in XA and XB using the distance metric 'abc'
    # but with a more succinct, verifiable, but less efficient implementation.

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')

    # The C code doesn't do striding.
    XA = _copy_array_if_base_present(_convert_to_double(XA))
    XB = _copy_array_if_base_present(_convert_to_double(XB))

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]
    dm = np.zeros((mA, mB), dtype=np.double)

    if isinstance(metric, string_types):
        mstr = metric.lower()

        try:
            validate, cdist_fn = _SIMPLE_CDIST[mstr]
            XA = validate(XA)
            XB = validate(XB)
            cdist_fn(XA, XB, dm)
            return dm
        except KeyError:
            pass
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm
