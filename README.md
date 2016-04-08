# simple-distance-cpy
Some of the functions from `scipy.spatial.distance` without tons of other code.

I needed only cdist function from `scipy.spatial.distance` and whole scipy was too much for Heroku,
so I extracted only this function from scipy.

Metric can be passed only as a string. Only *euclidean* metric was tested.
