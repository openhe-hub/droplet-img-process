## Dataset Format

- `id`:
  * type: int
  * frame id, also file number in one dataset folder
- `contour`: 
  * type: [[float, float]]
  * contour coordinates detected
- `area`:
  * type: float
  * area enclosed in the contour
- `circumstance`:
  * type: float
  * circumstance of the contour
- `circularity`:
  * type: float
  * circularity = 4 * pi * (area / circumstance ^ 2)
- `if_fell`:
  * type: bool
  * if the droplet have fell on the surface
- `velocity`:
  * type: float
  * the velocity of droplet spread on the surface
- `finger_num`:
  * type: int
  * the number of fingers
- `finger_centers`
  * type: [[float, float]]
  * the center coordinates of fingers
- `finger_lengths`
  * type: [float]
  * the lengths of fingers
- `circle_center`
  * type: [int, int]
  * the center of the regression circle
- `circle_radius`
  * type: int
  * the radius of the regression circle