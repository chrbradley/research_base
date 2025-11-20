# Irregular Polygon Packing Algorithm Research

## Problem Statement
Pack irregular polygons (stones) into an arbitrary target polygon (patio/walkway) while:
1. Respecting gap tolerance (e.g., 1" with 0.25" margin)
2. Allowing overlaps as cut lines (stones can be trimmed)
3. Distributing large and small stones evenly (hardscaping best practice)
4. Maximizing coverage

## Challenge Classification
- NP-hard optimization problem
- Similar to: 2D irregular bin packing, nesting problem, cutting stock problem
- Used in: Sheet metal cutting, leather cutting, textile industry, jigsaw puzzles

## Algorithm Approaches

### 1. Greedy Placement Algorithms

**Bottom-Left (BL) Algorithm:**
- Place each stone at lowest possible position, then leftmost
- Fast but often suboptimal
- Can be enhanced with rotations

**Bottom-Left-Fill (BLF):**
- Similar to BL but tries to fill gaps
- Better packing density

**Pros:**
- Fast (O(n²) to O(n³))
- Deterministic
- Simple to implement

**Cons:**
- Local optima
- No backtracking
- Quality depends on input order

### 2. Metaheuristic Algorithms

**Simulated Annealing:**
- Start with random/greedy placement
- Iteratively swap, move, rotate stones
- Accept worse solutions with decreasing probability
- Good balance of quality and speed

**Genetic Algorithm:**
- Encode placement as chromosome (position, rotation for each stone)
- Fitness: coverage, gap quality, size distribution
- Crossover: mix placements from two parents
- Mutation: perturb positions/rotations

**Particle Swarm Optimization:**
- Treat stones as particles
- Move toward best known positions
- Balance exploration and exploitation

**Pros:**
- Can escape local optima
- Flexible objective function
- Good solution quality

**Cons:**
- Slower than greedy
- Requires parameter tuning
- Non-deterministic

### 3. Physics-Based Simulation

**Rigid Body Simulation:**
- Model stones as rigid bodies with friction
- "Drop" stones into target area
- Let physics engine handle collision and settling
- Can use gravity + shaking simulation

**No-Fit Polygon (NFP) Method:**
- Pre-compute NFP for each stone pair
- NFP shows all positions where two polygons touch but don't overlap
- Enables fast collision detection

**Pros:**
- Intuitive and natural
- Handles complex shapes well
- Can incorporate real-world constraints

**Cons:**
- Computationally expensive
- Requires physics engine
- May need many iterations to converge

### 4. Hybrid Approaches

**Two-Phase Method:**
1. Rough placement (greedy or genetic)
2. Local improvement (simulated annealing or physics)

**Cluster-First, Pack-Second:**
1. Group stones by size
2. Place large stones first (greedy)
3. Fill gaps with smaller stones
4. Apply local optimization

### 5. Specialized Techniques

**Compaction Algorithms:**
- After initial placement, compact the layout
- Push stones toward each other to reduce gaps
- Maintain non-overlap constraint

**Beam Search:**
- Expand multiple partial solutions in parallel
- Keep top K candidates at each step
- Good for finding diverse solutions

## Collision Detection Libraries

**Shapely (Python):**
- Polygon operations: intersection, union, buffer
- Point-in-polygon, polygon-polygon intersection
- Built on GEOS (C++ library)

**PyGame:**
- Fast 2D collision detection
- Pixel-perfect collision masks

**Box2D (Python binding):**
- Physics engine with collision detection
- Rigid body simulation

**CGAL (C++ with Python bindings):**
- Computational geometry algorithms
- Exact arithmetic, robust

## Hardscaping Best Practices

Research on professional stone laying:

1. **Size Distribution:**
   - Mix large and small stones throughout
   - Avoid clustering same-size stones
   - "Rule of thirds": roughly equal distribution of small/medium/large

2. **Joint/Gap Spacing:**
   - Typical: 0.5" to 1" gaps for flagstone
   - Consistent gap width preferred
   - Gaps filled with polymeric sand or mortar

3. **Avoid Long Seams:**
   - Stagger joints (like brickwork)
   - Avoid continuous lines across patio
   - "Break the bonds"

4. **Edge Treatment:**
   - Larger stones on perimeter for stability
   - Avoid tiny slivers at edges

5. **Aesthetic Considerations:**
   - Color/texture distribution (future V2)
   - Grain direction (natural stone)
   - Visual balance

## Recommended Approach for V1

**Algorithm:** Greedy Placement with Local Optimization

**Phase 1: Initial Placement**
1. Sort stones by area (large to small)
2. Categorize as small/medium/large (by percentile)
3. Use round-robin from size categories to maintain distribution
4. For each stone:
   - Try multiple rotations (0°, 90°, 180°, 270°, and maybe 45° increments)
   - Try multiple positions (grid-based or edge-based)
   - Score each placement: coverage, gap quality, size distribution
   - Place at best position

**Phase 2: Gap Filling**
5. Identify remaining gaps
6. Try to fit remaining stones into gaps
7. Allow overlaps (marked as cut lines)

**Phase 3: Refinement**
8. Apply local swaps and adjustments
9. Optimize gap consistency
10. Final compaction

**Scoring Function:**
```
score = w1 * coverage_ratio
      - w2 * avg_gap_deviation
      - w3 * size_clustering_penalty
      - w4 * num_cuts
```

**Key Design Decisions:**
- Use Shapely for all polygon operations
- Target gap: 1" ± 0.25"
- Allow 5-10% overlap (cutting)
- Rotate stones in 15° or 30° increments
- Placement grid resolution: 1-2 inches

**Implementation Modules:**
1. `polygon_utils.py` - Polygon operations (Shapely wrappers)
2. `packing_algorithm.py` - Core packing logic
3. `scoring.py` - Placement scoring functions
4. `layout_optimizer.py` - Post-placement optimization

## Benchmarking Metrics

1. **Coverage:** Percentage of target area filled
2. **Gap consistency:** Std dev of gap widths
3. **Size distribution:** Gini coefficient or clustering metric
4. **Computation time:** Seconds to solution
5. **Number of cuts:** How many stones need trimming

## Future Optimizations (V2)

- Machine learning to predict good placements
- Parallel placement search (multi-threading)
- Interactive adjustment UI
- Constraint satisfaction solver (Z3, OR-Tools)
- Color/texture aware distribution
