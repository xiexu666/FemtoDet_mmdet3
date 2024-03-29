# FemtoDet-mmdet3
- change [FemtoDet](https://github.com/xiexu666/FemtoDet-deploy) into mmdetection 3.0

## train config
- configs in`configs/femtodet`
- train script `run.sh`
```
|  Detector  | Params | box AP50 |              Config                    | 
---------------------------------------------------------------------------
|            |        |   38.2   | ./configs/femtoDet/femtodet_0stage.py  |
                      -----------------------------------------------------
|  FemtoDet  | 68.77k |   39.8   | ./configs/femtoDet/femtodet_1stage.py  |
                      -----------------------------------------------------
|            |        |   43.6   | ./configs/femtoDet/femtodet_2stage.py  |
                      -----------------------------------------------------
|            |        |   45.7   | ./configs/femtoDet/femtodet_3stage.py  |
---------------------------------------------------------------------------
```