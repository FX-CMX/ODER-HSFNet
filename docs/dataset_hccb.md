# HCCB Dataset Notes

HCCB is the Highly Congested Classroom Behavior dataset introduced in the paper.

## Acquisition

- Scene: real multi-row tiered lecture classroom.
- Cameras: two Hikvision DS-2CD3T45-I3 HD bullet cameras deployed at the front-left and front-right sides of the classroom.
- Resolution: valid images standardized to 1920 x 1080.
- Size: 796 images, 50,229 student behavior boxes.
- Split: approximately 9:1 by image, with 45,000 training instances and 5,229 validation instances.

## Annotation Protocol

HCCB uses evidence-driven, priority-based, mutually exclusive single-label annotation. If one instance satisfies multiple behavior definitions, the behavior with the higher priority is selected.

| Priority | Class | Instances | Annotation criterion |
| ---: | --- | ---: | --- |
| 1 | sleeping | 201 | The upper body is clearly lying on the desk. |
| 2 | using phone | 4,395 | The student is operating a mobile phone. |
| 3 | writing | 2,327 | A writing action is visible, with the pen tip touching the paper. |
| 4 | reading | 7,492 | The student looks downward at a reading medium and the paper region is visible. |
| 5 | looking around | 4,883 | The student keeps the head raised and shows a non-forward attention shift. |
| 6 | heads up | 14,147 | The student keeps the head raised with a stable forward attention state. |
| 7 | bowing head | 16,784 | The student keeps the head lowered. |

## Class Distribution

| Class | Train | Val | Overall ratio |
| --- | ---: | ---: | ---: |
| reading | 6,655 | 837 | 14.92% |
| writing | 2,091 | 236 | 4.63% |
| heads up | 12,715 | 1,432 | 28.17% |
| sleeping | 182 | 19 | 0.40% |
| looking around | 4,383 | 500 | 9.72% |
| bowing head | 15,026 | 1,758 | 33.41% |
| using phone | 3,948 | 447 | 8.75% |
| total | 45,000 | 5,229 | 100.00% |

## Difficulty Structure

The paper characterizes HCCB through density, occlusion burden, and scale-depth conflict:

| Dataset | Images | Instances | Avg./Img. | P_img(30) | P_img(50) | P_obj(0.1) | R_F/B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SCB-D3-S | 5,015 | 25,806 | 5.15 | 0.74 | 0.00 | 64.28 | 4.95 |
| HCCB | 796 | 50,229 | 63.10 | 95.48 | 92.96 | 82.20 | 7.76 |

These statistics indicate that high-density samples dominate HCCB, object-level occlusion is widespread, and front-to-rear scale discontinuity is strong.
