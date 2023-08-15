:mod:`tiger.metrics`
====================

.. automodule:: tiger.metrics

Volume overlap
--------------

.. autofunction:: tiger.metrics.jaccard_score
.. autofunction:: tiger.metrics.dice_score
.. autofunction:: tiger.metrics.mean_dice_score

Surface distance
----------------

.. autofunction:: tiger.metrics.hausdorff_distance
.. autofunction:: tiger.metrics.average_surface_distance

Classification
--------------

.. autoclass:: tiger.metrics.BlandAltmanLimitsOfAgreement
    :members:

.. note::

    Cohen's kappa is not part of this library anymore, use sklearn instead.

Calcium scores
--------------

.. autofunction:: tiger.metrics.agatston_score
.. autofunction:: tiger.metrics.volume_score
.. autofunction:: tiger.metrics.mass_score
