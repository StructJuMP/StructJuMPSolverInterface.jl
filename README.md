StructJuMPSolverInterface
===

### StructJuMP's Solver Interface

StructJuMPSolverInterface defines the nonlinear solver interface for [StructJuMP](https://github.com/joehuchette/StructJuMP.jl)--the Structured Modeling Extension for [JuMP](https://github.com/JuliaOpt/JuMP.jl) and provides glue code for [PIPS](https://github.com/Argonne-National-Laboratory/PIPS) and [IPOPT](http://www.coin-or.org/Ipopt/documentation/) solvers.

PIPS parallel  interface ("PipsNlp") matches StructJuMP's paralel capabilities. The two offer a truly parallel modeling + solving environment. In addition, PIPS also has a serial interface ("PipsNlpSerial"), which is used mostly for debugging purposes. The Ipopt interface to StructJuMP is also serial.

### Solver Supports

<!--There are two solvers currently implement this interface. They are [PIPS](https://github.com/Argonne-National-Laboratory/PIPS) and [IPOPT](http://www.coin-or.org/Ipopt/documentation/). When using PIPS, user can choose either the parallel or serial implementations for solving the structured model. When the parallel solver is selected, the parallel problem allocation and generation are done automatically at the backend, that is also transparent to the user. It enables user to easily adopt the state-of-art parallel solvers and to solve large scale optimization problems which could be too big to allocate on a single node. -->



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

At this point, `m` is a two level model that has a single root node, or block and `3` children nodes. The model can be solved by calling `solve` function with a parameter `solver` equal to one of the known solvers, `"PipsNlp", "PipsNlpSerial"` or `"Ipopt"`. 
```julia
solve(m,solver="PipsNlp") #solving using parallel PIPS-NLP solver
```

### Exposed API functions
* `getLocalBlocksIds(m)` returns a vector of block IDs residing on the current MPI rank.   
* `getLocalChildrenIds(m)` returns a vector of children scenario IDs residing on the current MPI rank. 
```julia
@show getLocalChildrenIds(m) 
@show getLocalBlocksIds(m)
```

* `getModel(m,id)` returns the block with specified by `id`. The root block by default has the ID equals to 0. 
```julia
mm = getModel(m,1) # mm is now the 1st scenario node.
```

* `getVarValues(m,id)` returns the variable values vector for block `id`.
```julia
v = getVarValues(m,1) # v is holding the variable values of 1st scenario.
```

* `getVarValue(m,id,var_idx)` returns value of the variable indexed by `var_idx` in the block `id`.
```julia
a = getVarValue(m,1,2) #a is the 2nd variable value of block id # 1. 
```

* `getNumVars` and `getNumCons` return the number of variables and constraints of block `id`.  
* `getTotalNumVars` and `getTotalNumCons` return the total number of variables and constraints of the problem. The parameter `m` needs to point to the root block.
```julia
@show getNumVars(m,id)
@show getNumCons(m,id)
@show getTotalNumVars(m)
@show getTotalNumCons(m)
```

* `getObjectiveVal` returns the value of objective.
```julia
@show getObjectiveVal(m)
```

### Known Limitation 
* Variables in the structural blocks needs to be declared before the constraint declarations. 

<!-- [![Build Status](https://travis-ci.org/fqiang/SolverInterface.jl.svg?branch=master)](https://travis-ci.org/fqiang/SolverInterface.jl)
-->
