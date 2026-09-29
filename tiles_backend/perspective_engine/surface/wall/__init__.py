"""
Wall-specific behaviour.

Self-contained by design: nothing here imports from surface/floor/, so a wall
bug is fixed in this package and cannot regress the floor. The shared maths
comes from core/, camera/, geometry/ and lines/.

The one place a wall may CONSUME floor information is scale.py, which can take
a floor plane to borrow metric scale from. It arrives as a plain (a, b, c, d)
tuple handed in by the pipeline -- the wall never reaches into the floor
package to fetch one.
"""
