# Stone Segmentation Research

## Objective
Extract irregular polygon contours (vertices) from stone images for packing algorithm input.

## Approaches to Evaluate

### 1. SAM (Segment Anything Model) - Meta
**Versions:**
- SAM 1.0 (Original, 2023) - segment-anything package
- SAM 2 (2024) - Improved video segmentation, better boundary accuracy
- SAM 3 (2025) - Newly announced, need to verify availability

**Pros:**
- State-of-the-art zero-shot segmentation
- Excellent at handling irregular shapes
- Can work with prompts (points, boxes, masks)
- Pre-trained on massive dataset

**Cons:**
- Large model size (~2.4GB for ViT-H)
- Requires significant compute
- May be overkill for this specific task
- On-device deployment challenging

**Model Variants:**
- ViT-H (Huge): Best accuracy, ~2.4GB
- ViT-L (Large): Good balance, ~1.2GB
- ViT-B (Base): Faster, ~375MB
- Mobile-SAM: Lightweight variant (~40MB)

### 2. Traditional Computer Vision (OpenCV)
**Techniques:**
- Edge detection (Canny)
- Contour finding
- Adaptive thresholding
- Morphological operations
- GrabCut algorithm

**Pros:**
- Fast, lightweight
- No GPU required
- Predictable behavior
- Easy to debug and tune

**Cons:**
- Requires careful parameter tuning
- May struggle with complex textures
- Background separation challenging
- Less robust to varying conditions

### 3. Hybrid Approach
**Strategy:**
- Use SAM for initial segmentation
- Refine with traditional CV
- Extract clean polygon contours
- Simplify polygons for packing

### 4. Alternative Deep Learning Models

**U-Net variants:**
- Lightweight segmentation
- Can be fine-tuned for specific domain
- Good for stone/rock segmentation

**DeepLab:**
- Semantic segmentation
- Good boundary accuracy

**Mask R-CNN:**
- Instance segmentation
- Can detect multiple stones per image

**HuggingFace Models:**
- facebook/sam-vit-base
- facebook/sam-vit-large
- facebook/sam-vit-huge
- IDEA-Research/grounding-dino-base (for zero-shot detection)

## Evaluation Criteria

1. **Accuracy**: Clean boundary detection
2. **Speed**: Processing time per image
3. **Robustness**: Handles varying lighting, shadows
4. **Polygon Quality**: Smooth, simplified contours
5. **Deployment**: Model size, dependencies

## Testing Plan

1. Test SAM 2 (if available) or SAM 1.0 with different model sizes
2. Compare with OpenCV-based segmentation
3. Evaluate polygon extraction quality
4. Benchmark speed and resource usage
5. Select best approach for V1 prototype

## Initial Recommendation

For V1 research prototype:
- **Primary**: SAM 2 or SAM with ViT-B (base model) for balance of accuracy and speed
- **Fallback**: Traditional OpenCV if SAM is too slow or complex
- **Future**: Mobile-SAM or custom lightweight model for on-device processing

## Polygon Extraction Strategy

After segmentation:
1. Get binary mask from model
2. Find contours using OpenCV
3. Simplify polygon with Douglas-Peucker algorithm
4. Convert to real-world coordinates using ArUco calibration
5. Store as list of vertices for packing algorithm
