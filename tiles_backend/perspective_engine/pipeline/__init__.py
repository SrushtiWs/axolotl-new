"""
Per-surface render pipelines: floor_pipeline and wall_pipeline.

Migrated from the reference project. Its version of this file also imported
`pipeline/perspective_engine.estimate_perspective`, a legacy DeepLSD path that
needs PyTorch and that neither pipeline uses; that module was not migrated, so
this package imports cleanly without PyTorch.
"""
