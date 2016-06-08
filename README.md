StructJuMPSolverInterface
===

### StructJuMP's Solver Interface

StructJuMPSolverInterface defines the nonlinear solver interface for [StructJuMP](https://github.com/joehuchette/StructJuMP.jl)--the Structured Modeling Extension for [JuMP](https://github.com/JuliaOpt/JuMP.jl). StructJuMP provides user an easy way to model optimization problems using their inherited block structures. On the other hand, StructJuMPSolverInterface makes it easy for user to solve the structured model that is produced by StructJuMP using solvers that are implementing this interface. 


### Solver Supports

There are two solvers currently implement this interface. They are [PIPS](https://github.com/Argonne-National-Laboratory/PIPS) and [IPOPT](http://www.coin-or.org/Ipopt/documentation/). When using PIPS, user can choose either the parallel or serial implementations for solving the structured model. When the parallel solver is selected, the parallel problem allocation and generation are done automatically at the backend, that is also transparent to the user. It enables user to easily adopt the state-of-art parallel solvers and to solve large scale optimization problems which could be too big to allocate on a single node. 



### An Example

Declaring the root stage variables and constraints.
```julia
using StructJuMP, JuMP
using StructJuMPSolverInterface

scen = 3
m = StructuredModel(num_scenarios=scen)
@variable(m, x[1:2])
@NLconstraint(m, x[1] + x[2] == 100)
@NLobjective(m, Min, x[1]^2 + x[2]^2)
```

Declaring the second stage variables and constraints. 
```julia
for(i in getLocalChildrenIds(m))
    bl = StructuredModel(parent=m,id=i)
    @variable(bl, y[1:2])
    @NLconstraint(bl, x[1] + y[1]+y[2] ≥  0)
    @NLconstraint(bl, x[1] + y[1]+y[2]  ≤ 50)
    @NLobjective(bl, Min, y[1]^2 + y[2]^2)
end
```

At this point, `m` is a two level model that has a single root node and `3` children. Then the model can be solved by calling `solve` function with a input parameter `solver` that equals to one of the known solvers, `"PipsNlp", "PipsNlpSerial"` or `"Ipopt"`. 
```julia
solve(m,solver="PipsNlp") #solving using parallel PIPS-NLP solver
```

### Exposed API functions
* `getLocalChildrenIds(m)` returns a vector of children scenario IDs that is allocated on the calling process. `getLocalScenarioIds(m)` returns a vector of scenario IDs that is allocated in the calling process (include the root node).  
```julia
@show getLocalChildrenIds(m) 
@show getLocalScenarioIds(m)
```

* `getModel(m,id)` returns the node with scenario ID that equals to `id`. The root node by default has the ID equals to 0. 
```julia
mm = getModel(m,1) # mm is now the 1st scenario node.
```

* `getVarValues(m,id)` returns the variable values in a vector for the node with ID that equals to `id`.
```julia
v = getVarValues(m,1) # v is holding the variable values of 1st scenario.
```

* `getVarValue(m,id,var_idx)` returns a single variable value indexed by `var_idx` in the node with ID that equals to `id`.
```julia
a = getVarValue(m,1,2) #a is the 2nd variable value of 1st scenario. 
```

* `getNumVars` and `getNumCons` returns the number of variables and cnstraints correspondingly for a node with scenario ID that euqals to `id`.  `getTotalNumVars` and `getTotalNumCons` return the total number of variables and constraints by summing over every nodes.  
```julia
@show getNumVars(m,id)
@show getNumCons(m,id)
@show getTotalNumVars(m)
@show getTotalNumCons(m)
```

* `getObjectiveVal` returns the objective function value. 
```julia
@show getObjectiveVal(m)
```

### Known Limitation 
* Variables in sub-problem are assummed to be declared before the constraint declarations. 
* At this time, the exposed API functions are for serial solvers only. That is they computes the scenarios that are allocated on a local process. The API functions for enquiring model in a parallel allocation will be realsed shortly. 

<!-- [![Build Status](https://travis-ci.org/fqiang/SolverInterface.jl.svg?branch=master)](https://travis-ci.org/fqiang/SolverInterface.jl)
-->
