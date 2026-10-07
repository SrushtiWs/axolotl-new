# Graph Report - axolotl copy  (2026-10-07)

## Corpus Check
- 75 files · ~257,802 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 1 file(s) not represented in the graph (top: (none) 1)

## Summary
- 1093 nodes · 2489 edges · 57 communities (50 shown, 7 thin omitted)
- Extraction: 96% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 87 edges (avg confidence: 0.93)
- Token cost: 83,668 input · 0 output

## Community Hubs (Navigation)
- Calibrated Scene Assets
- Grounding DINO Detection
- Camera Estimation & Object Completion
- Extraction Config & Output
- Metric Tile Projection Engine
- Floor/Wall Detection & Cleanup
- Studio Page & Surface Selection
- Frontend API Client
- Frontend Type Contracts
- Strong Line Detection
- Committed Pipeline Assets & Reuse
- Frontend Build Dependencies
- Tiled Room Rendering
- Three.js Tile Layer
- Mirror Rejection & Extractor
- Matting & Layer Refinement
- Room Picker & Branding
- SAM2 Boundary Localisation
- LaMa Inpainting
- Grounding DINO Wrapper
- Mask Loading & Validation
- TypeScript Compiler Config
- Instance Selection
- Extraction Debug Overlays
- Room Data Import & Layout
- Segmentation API Endpoints
- Tile Render Response Assembly
- Room Profile Backups
- FastAPI App & Job Registry
- Render Geometry Detection
- Tile Visualizer Page
- Segment Core & Error Handling
- Perspective Geometry Math
- Clean Room & Analysis Panels
- DeepLSD Runner & Pipeline Stages
- Residual Object Handling
- Room Lookup & Manual Edits
- 3D View Shell & Limits
- Floor/Wall Design Rationale
- Tile Engine Contract
- Fitted vs Estimated Camera Paths
- Render Pipeline & three.json
- Object Extraction Rationale
- Room Profile Construction
- Result Download UI
- Pipeline Flow API
- Final Composite Clipping
- Image Upload Dropzone
- Profile Save & Validation
- three.json Record & Replay
- Layer Mask Writing
- Instance Overlap Resolution
- Profile Validation Rules
- Job-to-Room Save Locking
- Scene Runner Stubs
- Pipeline Info Component
- Vite Environment Types

## God Nodes (most connected - your core abstractions)
1. `ExtractionConfig` - 44 edges
2. `generate()` - 30 edges
3. `extract()` - 27 edges
4. `Studio()` - 26 edges
5. `Scene` - 24 edges
6. `render_tiled_room()` - 22 edges
7. `TileVisualizer()` - 20 edges
8. `write()` - 18 edges
9. `build()` - 17 edges
10. `compilerOptions` - 17 edges

## Surprising Connections (you probably didn't know these)
- `surfaceMaterial tile shader` --semantically_similar_to--> `engine.py metric tile projection`  [INFERRED] [semantically similar]
  frontend/src/3js/README.md → backend/README.md
- `Mask is the only clip` --semantically_similar_to--> `compositing.py final strict clip`  [INFERRED] [semantically similar]
  frontend/src/3js/README.md → backend/perspective_engine/README.md
- `roomConsistency() validation gate` --semantically_similar_to--> `masks.py mask loading and validation`  [INFERRED] [semantically similar]
  frontend/src/3js/README.md → backend/perspective_engine/README.md
- `Simulated progress in TileVisualizer.tsx` --conceptually_related_to--> `/src/main.tsx module entry`  [AMBIGUOUS]
  backend/README.md → frontend/index.html
- `backend render.py Rendered.three carrier` --conceptually_related_to--> `render.py render_tiled_room`  [INFERRED]
  frontend/src/3js/README.md → backend/perspective_engine/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Clean room to tiled room render flow** — backend_readme_lama_inpaint, backend_readme_clean_room, backend_readme_floor_wall_py, backend_perspective_engine_readme_masks_py, backend_perspective_engine_readme_render_py, backend_perspective_engine_readme_restore_props, backend_perspective_engine_readme_compositing_py [EXTRACTED 1.00]
- **Propose, localise, matte extraction stack** — backend_readme_segformer_proposals, backend_readme_sam_vitb, backend_readme_layers_refine_mask, backend_readme_matting_refine, backend_readme_union_before_matting [EXTRACTED 1.00]
- **three.json record and replay chain** — frontend_src_3js_readme_three_layer_py, frontend_src_3js_readme_floor_pipeline, frontend_src_3js_readme_wall_pipeline, frontend_src_3js_readme_render_py_three, frontend_src_3js_readme_save_three, frontend_src_3js_readme_three_json, frontend_src_3js_readme_threetiles [EXTRACTED 1.00]

## Communities (57 total, 7 thin omitted)

### Community 0 - "Calibrated Scene Assets"
Cohesion: 0.07
Nodes (23): health(), available(), image_signature(), load_scene(), _mask(), missing_assets(), _rgb(), scene_signature() (+15 more)

### Community 1 - "Grounding DINO Detection"
Cohesion: 0.07
Nodes (21): _chunk(), detect(), Detection, DinoUnavailable, download(), _load(), _phrase_spans(), _preprocess() (+13 more)

### Community 2 - "Camera Estimation & Object Completion"
Cohesion: 0.07
Nodes (17): MissingGeometryError, main(), build(), _clean_surface(), _far_edge_y(), _props_alpha(), _rotation(), _solve_pitch() (+9 more)

### Community 3 - "Extraction Config & Output"
Cohesion: 0.07
Nodes (16): CleanupConfig, DetectionConfig, OverlapConfig, ResidualConfig, SegmentationConfig, ExtractionResult, to_original(), __getattr__() (+8 more)

### Community 4 - "Metric Tile Projection Engine"
Cohesion: 0.11
Nodes (17): _contact_shadow(), _intersect_plane(), _intersect_plane_at(), _polynomial_illumination(), _project_floor(), _project_screeding(), _project_walls(), _rays() (+9 more)

### Community 5 - "Floor/Wall Detection & Cleanup"
Cohesion: 0.09
Nodes (15): clean(), _disk(), drop_small_components(), fill_small_holes(), smooth_contours(), _cut(), detect_floor_wall(), FloorWall (+7 more)

### Community 6 - "Studio Page & Surface Selection"
Cohesion: 0.08
Nodes (27): CompareSlider(), Props, Props, SurfaceMarker, SurfaceMarkers(), Props, Sort, TileRail() (+19 more)

### Community 7 - "Frontend API Client"
Cohesion: 0.10
Nodes (33): PipelineFlow(), STATUS_LABEL, DetectionList(), LayerCard(), SegmentationPanel(), SegmentCard(), Tab, ApiError (+25 more)

### Community 8 - "Frontend Type Contracts"
Cohesion: 0.10
Nodes (29): ProgressPanel(), ProgressPanelProps, FIELDS, RoomDimensions(), RoomDimensionsProps, Section(), SectionProps, TileSizeSelector() (+21 more)

### Community 9 - "Strong Line Detection"
Cohesion: 0.13
Nodes (15): _compose_layers(), compose_surfaces(), paint(), _load_layers(), _render_job(), _angle_diff(), _deeplsd(), _gradient_across() (+7 more)

### Community 10 - "Committed Pipeline Assets & Reuse"
Cohesion: 0.12
Nodes (11): _bbox(), _cut_out(), _localise(), object_layers(), _read_mask(), StagePreview, surfaces(), fit() (+3 more)

### Community 11 - "Frontend Build Dependencies"
Cohesion: 0.07
Nodes (27): dependencies, react, react-dom, three, devDependencies, @types/react, @types/react-dom, @types/three (+19 more)

### Community 12 - "Tiled Room Rendering"
Cohesion: 0.11
Nodes (12): describe(), _detected_horizon_vp(), _dot(), _horizon_vp(), _instance(), render_tiled_room(), Rendered, _room_box_mm() (+4 more)

### Community 13 - "Three.js Tile Layer"
Cohesion: 0.15
Nodes (21): jobBase(), ThreeTileLayer(), ThreeTileLayerProps, agrees(), coverageGeometry(), exactTexture(), jobCache, loadThreeJob() (+13 more)

### Community 14 - "Mirror Rejection & Extractor"
Cohesion: 0.10
Nodes (11): _bounds(), extract(), ExtractedObject, _fit_within(), structural_map(), classify(), complete(), group() (+3 more)

### Community 15 - "Matting & Layer Refinement"
Cohesion: 0.17
Nodes (11): compose(), _disk(), _drop_small_components(), _fill_small_holes(), refine_mask(), _box(), decontaminate(), _disk() (+3 more)

### Community 16 - "Room Picker & Branding"
Cohesion: 0.12
Nodes (18): App(), Brand(), Props, SegmentationPanelProps, container, Props, RoomPicker(), useUpload() (+10 more)

### Community 17 - "SAM2 Boundary Localisation"
Cohesion: 0.14
Nodes (10): available(), download(), Embedding, encode(), _feeds(), _load(), logits_at_full_resolution(), masks_for_boxes() (+2 more)

### Community 18 - "LaMa Inpainting"
Cohesion: 0.13
Nodes (9): available(), clean_room(), _fill(), grow_px(), hole_from_layers(), InpaintUnavailable, _load(), verify() (+1 more)

### Community 19 - "Grounding DINO Wrapper"
Cohesion: 0.14
Nodes (6): ExtractionConfig, detect_and_segment(), detect_to_dicts(), main(), run_independent(), run_with_masks()

### Community 20 - "Mask Loading & Validation"
Cohesion: 0.15
Nodes (7): _binary(), load(), MaskError, prepare(), _to_shape(), ensure_geometry(), _objects_mask()

### Community 21 - "TypeScript Compiler Config"
Cohesion: 0.11
Nodes (18): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleDetection, moduleResolution (+10 more)

### Community 22 - "Instance Selection"
Cohesion: 0.18
Nodes (8): choose(), _components(), _failures(), _measure(), _rank(), Selected, trim_surfaces(), validate_final()

### Community 23 - "Extraction Debug Overlays"
Cohesion: 0.26
Nodes (9): _canvas(), _checkerboard(), _colour(), _label(), _mask_overlay(), _over_checkerboard(), _save(), write() (+1 more)

### Community 24 - "Room Data Import & Layout"
Cohesion: 0.18
Nodes (8): digest(), ensure_layout(), images_in_folder(), import_rooms(), _next_room_id(), now_iso(), save_index(), sniff_format()

### Community 25 - "Segmentation API Endpoints"
Cohesion: 0.18
Nodes (5): _decode(), floor_wall_masks(), room_surfaces(), segment_room(), segment_start()

### Community 26 - "Tile Render Response Assembly"
Cohesion: 0.12
Nodes (9): _clean_for_render(), generate(), analyse(), _legacy_room_mm(), _positive(), _save_three(), _tile_engine_notes(), _wall_note() (+1 more)

### Community 27 - "Room Profile Backups"
Cohesion: 0.24
Nodes (10): _backup_current(), _backup_dir(), _backups(), _copy_dir(), load_profile(), original_path(), _preview(), room_dir() (+2 more)

### Community 29 - "Render Geometry Detection"
Cohesion: 0.20
Nodes (7): _accepted_union(), _detect_floor_wall(), _detect_geometry(), _render_clean(), _save_layers(), analyse(), _to_shape()

### Community 30 - "Tile Visualizer Page"
Cohesion: 0.23
Nodes (14): Option, OptionGroup(), OptionGroupProps, resolveTileSize(), TileVisualizer(), handleGenerate(), handleReset(), release() (+6 more)

### Community 31 - "Segment Core & Error Handling"
Cohesion: 0.14
Nodes (7): _fit_within(), _geometry_urls(), _segment_core(), run(), load(), ExtractionError, digest()

### Community 32 - "Perspective Geometry Math"
Cohesion: 0.24
Nodes (5): exists(), _json(), load(), _optional_json(), write()

### Community 33 - "Clean Room & Analysis Panels"
Cohesion: 0.27
Nodes (11): CleanRoom(), compose(), load(), Props, Props, RoomAnalysis(), clearRoom(), segmentRoomAsync() (+3 more)

### Community 34 - "DeepLSD Runner & Pipeline Stages"
Cohesion: 0.18
Nodes (4): flow(), _load_definition(), demo_rooms(), write_json_atomic()

### Community 35 - "Residual Object Handling"
Cohesion: 0.27
Nodes (5): _best_candidate(), _bounds(), regions(), residual_map(), sweep()

### Community 36 - "Room Lookup & Manual Edits"
Cohesion: 0.18
Nodes (6): room_data_edit(), room_data_room(), _room_entry(), find_room(), load_index(), saved_response()

### Community 37 - "3D View Shell & Limits"
Cohesion: 0.20
Nodes (7): engine._restore_props objects back on top, frontend index.html app shell, /src/main.tsx module entry, 3js/index.ts public exports, 3js Three.js tile layer, pages/Studio.tsx 3D View button, ThreeTileLayer.tsx stacking component

### Community 38 - "Floor/Wall Design Rationale"
Cohesion: 0.22
Nodes (8): Band-limited GrabCut boundary refinement, POST /floor-wall endpoint, floor_wall.py two-class surface detection, layers.refine_mask union cleanup, matting.refine and matting.decontaminate, surfaces.py ADE20K semantic segmenter, numpy>=1.26, opencv-python-headless

### Community 39 - "Tile Engine Contract"
Cohesion: 0.27
Nodes (9): compositing.py final strict clip, masks.py mask loading and validation, backend/perspective_engine package, render.py render_tiled_room, tiles_backend/perspective_engine reference tile engine, Wall choice and wall_region, roomConsistency() validation gate, RoomGeometry canonical room (Phase 3) (+1 more)

### Community 40 - "Fitted vs Estimated Camera Paths"
Cohesion: 0.24
Nodes (8): Parity against committed pipeline output, engine.py metric tile projection, estimated path - synthesised camera, fitted path - calibrated room camera, live_scene.py estimated camera synthesis, pipeline_assets.py committed stage 06F4 cut-outs, segmentation.py fast single-model path, Two geometry paths (fitted vs estimated)

### Community 41 - "Render Pipeline & three.json"
Cohesion: 0.27
Nodes (9): The clean room (emptied room render), POST /generate tile render endpoint, backend/jobs/<job_id> output layout, extraction/inpaint.py LaMa windowed fill, fastapi + uvicorn + python-multipart, projectionFromIntrinsics, app.py _save_three and three_url, three.json render state record (+1 more)

### Community 42 - "Object Extraction Rationale"
Cohesion: 0.24
Nodes (6): precise.py propose-localise-matte extraction, SAM ViT-B boundary localisation, SegFormer-B4 ADE20K proposals as SAM prompts, POST /segment object extraction endpoint, onnxruntime (CPU inference runtime), tokenizers (Grounding DINO text side)

### Community 43 - "Room Profile Construction"
Cohesion: 0.22
Nodes (5): build_profile(), _near_parallel_pairs(), _outlines(), _read_mask(), _side()

### Community 44 - "Result Download UI"
Cohesion: 0.29
Nodes (7): DownloadButton(), DownloadButtonProps, State, ResultPanel(), ResultPanelProps, GenerateResponse, GenerationStatus

### Community 45 - "Pipeline Flow API"
Cohesion: 0.25
Nodes (3): pipeline_flow(), pipeline_segments(), room_data_index()

### Community 46 - "Final Composite Clipping"
Cohesion: 0.28
Nodes (3): Result, allowed(), clip()

### Community 47 - "Image Upload Dropzone"
Cohesion: 0.33
Nodes (7): formatSize(), ImageDropzone(), accept(), handleDrop(), ImageDropzoneProps, RoomIcon(), TileIcon()

### Community 48 - "Profile Save & Validation"
Cohesion: 0.32
Nodes (4): ProfileError, save_profile(), _update_index(), update_profile()

### Community 49 - "three.json Record & Replay"
Cohesion: 0.29
Nodes (6): floor_pipeline.py info['three'], backend render.py Rendered.three carrier, surfaceMaterial tile shader, three_layer.py record(), three_parity.py 3D-vs-2D verification, wall_pipeline.py per-wall 'three' entry

### Community 52 - "Profile Validation Rules"
Cohesion: 0.40
Nodes (3): dot_problems(), _is_point(), validate_profile()

## Ambiguous Edges - Review These
- `precise.py propose-localise-matte extraction` → `tokenizers (Grounding DINO text side)`  [AMBIGUOUS]
  backend/requirements.txt · relation: conceptually_related_to
- `Simulated progress in TileVisualizer.tsx` → `/src/main.tsx module entry`  [AMBIGUOUS]
  backend/README.md · relation: conceptually_related_to

## Knowledge Gaps
- **88 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+83 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 449 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `precise.py propose-localise-matte extraction` and `tokenizers (Grounding DINO text side)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `ExtractionConfig` connect `Grounding DINO Wrapper` to `Grounding DINO Detection`, `Extraction Config & Output`, `Residual Object Handling`, `Floor/Wall Detection & Cleanup`, `Mirror Rejection & Extractor`, `Instance Overlap Resolution`, `Instance Selection`, `Segment Core & Error Handling`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Are the 25 inferred relationships involving `ExtractionConfig` (e.g. with `clean()` and `smooth_contours()`) actually correct?**
  _`ExtractionConfig` has 25 INFERRED edges - model-reasoned connections that need verification._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _88 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Calibrated Scene Assets` be split into smaller, more focused modules?**
  _Cohesion score 0.06561085972850679 - nodes in this community are weakly interconnected._
- **What is the exact relationship between `Simulated progress in TileVisualizer.tsx` and `/src/main.tsx module entry`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `react` connect `Clean Room & Analysis Panels` to `Studio Page & Surface Selection`, `Frontend API Client`, `Frontend Type Contracts`, `Frontend Build Dependencies`, `Result Download UI`, `Three.js Tile Layer`, `Image Upload Dropzone`, `Room Picker & Branding`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._