# Methods

## Backend

|Backend|Description|Advantage|Disadvantage|
| ---- | ---- | ---- | ---- |
|**Python Function**|The function developed by this project incorporates the latest trajectory inference methods in recent years, making it particularly well-suited to the project's framework.|1. New Methods in recenty years. <br>| 1. Different trajectory inference package versions in the same Python environment may conflict. |
|**Dynverse Docker**|Docker image for trajectory inference refers to dynverse [@dynverse].|1. The ease of use of Docker |1.Methods on R language not be compatible. <br> 2.Methods are old relatively. <br> 3. Docker environment is need.
|**CFE Docker**|Docker image for trajectory inference are developed by this project.|1. New Methods in recenty years. <br>2. The ease of use of Docker. |1. Docker environment is need. |

## Reference source

- **Dynverse**[@dynverse]: 45 methods filtered from 70 methods before 2019 years. Output results of them can be classfied to 7 wrapper. Paper, github reository, document are available.
- **Github Reporsitory**[@sc_pseudotime_github]: A repository keeps track of the latest trajectory infernce methods in real-time. Related topics such as upstream opertion(data imputation, dimsional reduction), donstream analysis(GRN inference, trajectory alignment) and reviews are also included.

## Implementation order (TODO List)

- [ ] Dynverse represtive methods for 7 basic wrapper:
    - [x] `Direct`: PAGA
    - [x] `Linear`: Component 1(baseline)
    - [ ] `Cycle`: Angle(baseline)
    - [ ] `Prob`: SCUBA
    - [ ] `Cluster`: GrandPrix
    - [x] `Proj`: MST(baseline)
    - [ ]  `Cell`: Monocle2
- [ ] `Velocity` wrapper:
    - [ ] Strategy from `Velocity` wrapper to `Direct` wrapper.
    - [ ] represtive method scVelo.
- [ ] For other methods, the higher the citation count of the paper, the higher the implementation order(need statistics from google scholar).

> The work integrate trajectory methods from the issue area continuously.
