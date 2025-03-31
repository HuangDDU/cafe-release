# Methods

## 1. Backend

|Backend|Description|Advantage|Disadvantage|
| ---- | ---- | ---- | ---- |
|**Python Function**|The function developed by this project incorporates the latest trajectory inference methods in recent years, making it particularly well-suited to the project's framework.|1. New Methods in recenty years. <br>| 1. Different trajectory inference package versions in the same Python environment may conflict. |
|**Dynverse Docker**|Docker image for trajectory inference refers to dynverse [@dynverse].|1. The ease of use of Docker |1.Methods on R language not be compatible. <br> 2.Methods are old relatively. <br/> 3. Docker environment is need.
|**CFE Docker**|Docker image for trajectory inference are developed by this project.|1. New Methods in recenty years. <br>2. The ease of use of Docker. |1. Docker environment is need. |

## Reference source

- **Dynverse**[@dynverse]: 45 methods filtered from 70 methods before 2019 years. Output results of them can be classfied to 7 wrapper. Paper, github reository, document are available.
- **Github Reporsitory**[@sc_pseudotime_github]: A repository keeps track of the latest trajectory infernce methods in real-time. Related topics such as upstream opertion(data imputation, dimsional reduction), donstream analysis(GRN inference, trajectory alignment) and reviews are also included.

## 2.Implementation order

### 2.1 wrapper and baseline(Completed on 2023.03.11)

> Here, baseline methods are easy way to get the specified aimed wrapper input data structure, where MST(Minimum Spanning Tree) are widely used.

- [x] Dynverse represtive methods for 7 basic wrapper:
    - [x] `Direct`: PAGA
    - [x] `Linear`: Component 1(baseline)
    - [x] `Cycle`: Angle(baseline)
    - [x] `Probability`: State Component(baseline)
    - [x] `Cluster`: Cluster MST(baseline)
    - [x] `Projection`: Projection MST(baseline)
    - [x]  `Graph`: Graph MST(baseline)
- [x] `Velocity` wrapper:
    - [x] Strategy from `Velocity` wrapper to `Direct` wrapper.
    - [x] represtive method scVelo.
- [] CFE Docker:
    - [x]Use Docker to manage environments and version of specific methods.
    - [] Use Github Action to build and push docker images automatically, (now, the action script is triggerd manually).

### 2.2 More published methods (Working)
> ref: https://github.com/agitter/single-cell-pseudotime

- For other methods, the higher the citation count of the paper, the higher the implementation order(need statistics from google scholar).

| Wrapper Type | Method Name |
| --- | --- |
| Direct | PAGA |
| Linear | Component 1(baseline) |
|  | Palantir |
|  | Cytotrace/Cytotrace2 |
| Cycle | Angle(baseline) |
| Probability | State Component(baseline) |
|  | CellRank |
| Cluster | Cluster MST(baseline) |
| Projection | Projection MST(baseline)|
| Graph | Graph MST(baseline) |
| Velocity | scVelo|
|  | Dynamo |
|  | VeloAE |

> Methods to be categorized: WaddingtonOT, TrajectoryNet, pyVIA, 

### TODO
> The work integrate trajectory methods from the issue area continuously.
