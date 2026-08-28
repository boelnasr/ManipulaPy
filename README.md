# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/boelnasr/ManipulaPy/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                            |    Stmts |     Miss |   Branch |   BrPart |     Cover |   Missing |
|------------------------------------------------ | -------: | -------: | -------: | -------: | --------: | --------: |
| ManipulaPy/backend/jax\_backend.py              |      154 |      154 |       26 |        0 |      0.0% |    38-384 |
| ManipulaPy/backend/torch\_backend.py            |      329 |      329 |      114 |        0 |      0.0% |    38-729 |
| ManipulaPy/urdf/visualization/pybullet\_viz.py  |        8 |        8 |        0 |        0 |      0.0% |      9-46 |
| ManipulaPy/urdf/visualization/trimesh\_viz.py   |       95 |       86 |       54 |        0 |      6.0% |22-24, 33-66, 75-112, 121-122, 132-189 |
| ManipulaPy/urdf/geometry/primitives.py          |       59 |       53 |        8 |        0 |      9.0% |27-67, 87-125, 143-199, 206-241 |
| ManipulaPy/urdf/geometry/mesh\_loader.py        |      104 |       87 |       46 |        0 |     11.3% |45-66, 75-112, 117-143, 148-185, 193-195 |
| ManipulaPy/urdf/visualization/\_\_init\_\_.py   |       22 |       18 |        0 |        0 |     18.2% |30-47, 63-80 |
| ManipulaPy/cuda\_kernels/registry.py            |      337 |      236 |       94 |        4 |     24.8% |95-117, 127-130, 132-135, 203-260, 281-332, 340-342, 358-676, 689, 693, 756-783, 803-809, 820, 878-880 |
| ManipulaPy/planning/benchmarks.py               |       90 |       61 |       30 |        3 |     28.3% |59-69, 89-94, 120, 136, 182-254, 279-338 |
| ManipulaPy/urdf/scene.py                        |      157 |      101 |       52 |        1 |     29.2% |45-46, 51-52, 110, 137-141, 153-155, 186-199, 248-261, 276-305, 320-350, 368-389, 408-418, 443-473, 493-511 |
| ManipulaPy/cuda\_kernels/memory.py              |       49 |       33 |        8 |        1 |     29.8% |35-45, 55-143, 162 |
| ManipulaPy/cuda\_kernels/trajectory\_kernels.py |      161 |      103 |       28 |        1 |     34.4% |92-1081, 1130, 1236-1288 |
| ManipulaPy/urdf/xacro.py                        |       78 |       45 |       18 |        1 |     37.5% |54-76, 107-\>133, 167-185, 213-269, 282-283 |
| ManipulaPy/urdf\_processor.py                   |      149 |       81 |       26 |        2 |     40.0% |107-111, 130-131, 146, 154, 162-168, 211-218, 237-262, 276, 313, 329-357, 370-371, 383, 395, 414, 433, 459-468, 482-483, 500, 521-532, 539, 544, 549, 554, 564, 583, 601, 610-611, 652, 677 |
| ManipulaPy/\_\_init\_\_.py                      |      146 |       76 |       62 |        6 |     40.4% |59-65, 221-\>226, 269-\>271, 271-\>273, 273-\>276, 303-328, 338-339, 349-350, 367-375, 391-408, 419-473, 478, 484-\>exit |
| ManipulaPy/urdf/modifiers.py                    |      296 |      148 |      142 |       29 |     44.1% |85-98, 115-126, 155, 157-\>159, 159-\>161, 162, 164, 186-212, 237, 258-268, 285-296, 319-340, 361, 364-\>359, 401-405, 426, 440-444, 459-473, 486-510, 527-533, 540-\>557, 576-\>587, 581-\>583, 583-\>585, 587-\>595, 596, 606-624, 646, 650-\>654, 654-\>exit, 655-\>exit, 664, 668-\>exit, 689-695, 701-\>703, 703-\>705, 707-\>exit, 710-\>exit, 720-721, 748-766, 780-795 |
| ManipulaPy/planning/\_plotting.py               |      123 |       65 |       20 |        3 |     45.5% |51-56, 62, 98-146, 172-176, 218-297 |
| ManipulaPy/planning/trajectory\_dynamics.py     |      185 |       81 |       44 |        5 |     56.3% |70, 76, 122-306, 411, 417, 451-578, 671-677 |
| ManipulaPy/backend/cupy\_backend.py             |       85 |       36 |        0 |        0 |     57.6% |56, 59, 62, 65, 68, 71, 75, 78, 81, 84, 87, 92, 95, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 136, 139, 142, 145, 148, 151, 154, 157, 161, 167, 171 |
| ManipulaPy/urdf/validation.py                   |      126 |       40 |       64 |       18 |     60.0% |43-48, 66-75, 80, 85, 89-97, 140-144, 156, 164, 173, 184-\>186, 193, 198, 205-\>216, 210, 218, 229-230, 246, 271-272, 294, 302-\>319, 304, 311, 320-321, 336-355 |
| ManipulaPy/planning/trajectory\_planning.py     |      236 |       73 |       82 |       19 |     66.7% |103-\>105, 108-110, 263-266, 274-276, 296, 304-\>311, 318, 384, 388-394, 421, 423, 428, 433, 450, 469, 479-483, 508-509, 542-643, 674, 742-756, 765 |
| ManipulaPy/vision.py                            |      317 |      101 |       74 |       10 |     67.0% |113-129, 154-165, 188-190, 342-347, 409-415, 543-560, 582, 593-\>607, 596, 695-696, 717-718, 724, 738, 741, 748, 782-835, 845-856, 864-882, 888-902, 915-918, 928-929 |
| ManipulaPy/cuda\_kernels/field\_kernels.py      |       55 |       18 |        6 |        2 |     67.2% |18, 149, 173-218 |
| ManipulaPy/cuda\_kernels/\_runtime.py           |       89 |       26 |       36 |        5 |     68.8% |50-71, 97-159, 331, 341-\>exit, 352, 491-\>474 |
| ManipulaPy/urdf/core.py                         |      351 |       97 |       96 |       17 |     72.3% |69-\>75, 133, 159-287, 301-\>311, 308, 343, 349, 355, 370-372, 378, 385, 397-399, 410, 416, 424, 432-\>436, 465, 484-\>483, 493-\>492, 510, 566, 596, 619-620, 624, 631, 652-666, 833-835, 851-853, 859, 863, 877-890, 894-896, 909-917 |
| ManipulaPy/urdf/types.py                        |      317 |       70 |       34 |        5 |     75.2% |41-47, 152-166, 171, 175-177, 181, 251-254, 276-278, 283, 295-307, 325-327, 331, 370, 383, 403-406, 417-419, 423, 440-\>exit, 445-447, 455-456, 493, 567, 571-573, 598-600, 723-729, 766, 775-780, 845, 849-851 |
| ManipulaPy/planning/trajectory.py               |      246 |       51 |       64 |       12 |     75.8% |39-75, 99, 131-\>134, 223, 230-233, 239, 251, 352-\>360, 381, 394-417, 578, 640, 669-\>671, 671-\>673, 717 |
| ManipulaPy/sim/rendering.py                     |      112 |       23 |       34 |        7 |     76.7% |47, 54, 70-72, 136, 248-274, 286-\>exit, 324-341 |
| ManipulaPy/urdf/parser.py                       |      296 |       45 |      120 |       28 |     78.6% |89-91, 95-\>99, 101-102, 103-\>106, 152, 158-\>156, 207-221, 261, 278-291, 295, 308-312, 320, 367, 426, 454-\>458, 461, 473, 489, 495-\>499, 502, 729-732, 839, 846, 851-\>854, 866, 871-\>878 |
| ManipulaPy/urdf/resolver.py                     |      312 |       46 |      166 |       23 |     81.4% |94-95, 114-\>112, 119-122, 130-\>128, 136-158, 176, 186-\>exit, 188-\>190, 194-\>exit, 214, 242, 415, 418, 425, 428, 433-434, 467-468, 486-487, 503-\>508, 518-521, 538, 575-581, 597, 605-606, 620-621, 643, 652 |
| ManipulaPy/cuda\_kernels/\_\_init\_\_.py        |       17 |        2 |        2 |        1 |     84.2% |     90-91 |
| ManipulaPy/sim/simulation.py                    |      243 |       31 |       44 |        8 |     84.3% |86, 141, 143, 161-\>163, 211-223, 241-244, 457, 459-461, 510-511, 554-555, 584-\>589, 591-604, 607-608 |
| ManipulaPy/kinematics/ik.py                     |      240 |       19 |       88 |       18 |     88.1% |118-124, 174-\>176, 176-\>173, 267, 282, 394, 405, 412, 437, 459, 475, 530, 540-543, 561, 579, 584-\>526, 596, 646-\>649 |
| ManipulaPy/backend/\_\_init\_\_.py              |       69 |        6 |       20 |        4 |     88.8% |87, 89, 118, 137, 149, 169 |
| ManipulaPy/singularity/singularity\_analysis.py |       82 |        9 |       10 |        1 |     89.1% |229-244, 286 |
| ManipulaPy/sim/\_runtime.py                     |       35 |        2 |       12 |        3 |     89.4% |60-\>62, 69, 75 |
| ManipulaPy/planning/\_kernels.py                |       35 |        2 |        2 |        1 |     91.9% |     66-68 |
| ManipulaPy/sim/controllers.py                   |       63 |        3 |       16 |        3 |     92.4% |32-39, 61-\>exit, 86-\>99, 96 |
| ManipulaPy/perception.py                        |       70 |        3 |       14 |        3 |     92.9% |216-217, 256, 279-\>exit |
| ManipulaPy/kinematics/serial\_manipulator.py    |       47 |        2 |       10 |        2 |     93.0% |  105, 113 |
| ManipulaPy/kinematics/trac\_ik.py               |      271 |       10 |       76 |       10 |     94.2% |167-\>exit, 207-\>220, 211-\>207, 221-235, 248-\>264, 265-273, 278, 343, 371-\>499, 562 |
| ManipulaPy/control/metrics.py                   |      123 |        3 |       26 |        4 |     95.3% |175, 179-\>183, 184, 346 |
| ManipulaPy/planning/collision\_host.py          |       46 |        1 |       20 |        2 |     95.5% |76-\>86, 143 |
| ManipulaPy/potential\_field/collision.py        |       97 |        2 |       38 |        4 |     95.6% |99-\>102, 107, 114-\>116, 130 |
| ManipulaPy/kinematics/ik\_helpers.py            |      120 |        1 |       48 |        6 |     95.8% |68-\>72, 87-\>89, 89-\>91, 91-\>104, 95-\>97, 446 |
| ManipulaPy/backend/numpy\_backend.py            |       86 |        3 |        0 |        0 |     96.5% |143, 159, 165 |
| ManipulaPy/dynamics/forces.py                   |       58 |        2 |       10 |        0 |     97.1% |     23-24 |
| ManipulaPy/control/computed\_torque.py          |       78 |        1 |        6 |        1 |     97.6% |        65 |
| ManipulaPy/control/manipulator\_controller.py   |       77 |        1 |       10 |        1 |     97.7% |       112 |
| ManipulaPy/utils/so3.py                         |       98 |        1 |        2 |        1 |     98.0% |       107 |
| ManipulaPy/control/\_\_init\_\_.py              |       51 |        0 |       14 |        1 |     98.5% | 124-\>126 |
| ManipulaPy/dynamics/mass\_matrix.py             |       56 |        0 |       16 |        1 |     98.6% | 129-\>132 |
| ManipulaPy/utils/se3.py                         |       87 |        1 |        0 |        0 |     98.9% |        24 |
| ManipulaPy/backend/base.py                      |        3 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/control/kalman.py                    |       51 |        0 |        4 |        0 |    100.0% |           |
| ManipulaPy/control/pid.py                       |       38 |        0 |        6 |        0 |    100.0% |           |
| ManipulaPy/control/robust\_adaptive.py          |       41 |        0 |        2 |        0 |    100.0% |           |
| ManipulaPy/dynamics/\_\_init\_\_.py             |        9 |        0 |        2 |        0 |    100.0% |           |
| ManipulaPy/dynamics/cache.py                    |       26 |        0 |        8 |        0 |    100.0% |           |
| ManipulaPy/dynamics/id\_fd.py                   |       22 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/dynamics/manipulator\_dynamics.py    |       25 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/ik\_helpers.py                       |        1 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/kinematics/\_\_init\_\_.py           |       11 |        0 |        2 |        0 |    100.0% |           |
| ManipulaPy/kinematics/fk.py                     |       33 |        0 |        8 |        0 |    100.0% |           |
| ManipulaPy/kinematics/jacobian.py               |       30 |        0 |       10 |        0 |    100.0% |           |
| ManipulaPy/kinematics/velocity.py               |       19 |        0 |        4 |        0 |    100.0% |           |
| ManipulaPy/path\_planning.py                    |        3 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/planning/\_\_init\_\_.py             |        6 |        0 |        2 |        0 |    100.0% |           |
| ManipulaPy/potential\_field/\_\_init\_\_.py     |        4 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/potential\_field/adjacency.py        |       12 |        0 |        8 |        0 |    100.0% |           |
| ManipulaPy/potential\_field/fields.py           |       78 |        0 |        6 |        0 |    100.0% |           |
| ManipulaPy/sim/\_\_init\_\_.py                  |        4 |        0 |        2 |        0 |    100.0% |           |
| ManipulaPy/singularity/\_\_init\_\_.py          |        7 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/trac\_ik.py                          |        1 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/urdf/\_\_init\_\_.py                 |        8 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/urdf/geometry/\_\_init\_\_.py        |        3 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/utils/\_\_init\_\_.py                |        4 |        0 |        0 |        0 |    100.0% |           |
| ManipulaPy/utils/screw.py                       |       53 |        0 |       14 |        0 |    100.0% |           |
| ManipulaPy/utils/time\_scaling.py               |        4 |        0 |        0 |        0 |    100.0% |           |
| **TOTAL**                                       | **7729** | **2496** | **2110** |  **277** | **64.3%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/boelnasr/ManipulaPy/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/boelnasr/ManipulaPy/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/boelnasr/ManipulaPy/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/boelnasr/ManipulaPy/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fboelnasr%2FManipulaPy%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/boelnasr/ManipulaPy/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.