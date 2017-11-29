# orbpy
Simple orbit propagation in Python

I wrote this (very quickly) in support of
[estpy](http://www.github.com/rwilson4/estpy). It's not intended to be
used by people who actually want a propagator. Folks who want a real
propagator can have a look at
[poliastro](http://docs.poliastro.space/en/latest/index.html). The
python propagators I've seen seem to focus on sun-centered systems,
like inter-planetary missions. Consequently, they don't seem to focus
on Earth-centric propagation, in which the effects of the Sun and
Moon's gravity, and the non-spherical mass distribution of the Earth
are important. If anyone can point me to a propagator that can take
into account the non-spherical Earth, Sun/Moon effects, drag, light
radiation, etc., let me know!

