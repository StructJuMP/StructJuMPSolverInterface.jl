#
# front-end for StructJuMP solvers interfaces
#

module StructJuMPSolverInterface


# Struct Model interface
abstract ModelInterface

export ModelInterface, KnownSolvers, sj_solve, getModel, getVarValue, getVarValues, getNumVars, getNumCons, getTotalNumVars, getTotalNumCons, getLocalScenarioIDs, getLocalChildrenIds

KnownSolvers = Dict{AbstractString,Function}();

function sj_solve(model; solver="Unknown", with_prof=false, suppress_warmings=false,kwargs...)
    if !haskey(KnownSolvers,solver)
        Base.warn("Unknow solver: ", solver)
        Base.error("Known solvers are: ", keys(KnownSolvers))
    end
    KnownSolvers[solver](model; with_prof=with_prof, suppress_warmings=false,kwargs...)
end

# package code goes here
include("helper.jl")
include("structure_helper.jl")
include("nonstruct_helper.jl")

function getModel(m,id)
    return id==0?m:getchildren(m)[id]
end

function getVarValues(m,id)
    mm = getModel(m,0); v = Float64[];
    for i = 1:getNumVars(m,id)
        v = [v;JuMP.getvalue(JuMP.Variable(mm,i))]
    end
    return v
end

function getVarValue(m,id,idx)
    mm = getModel(m,id)
    @assert idx<=getNumVars(m,id)
    return JuMP.getvalue(JuMP.Variable(mm,idx))
end

##!feng function getObjectiveValue(m)
##! It seems that getobjectivevalue(m) does not return the correct objetive value
##!feng end

function getNumVars(m,id)
  mm = getModel(m,id)
  nvar = MathProgBase.numvar(mm) - length(getStructure(mm).othermap)
  return nvar
end

function getNumCons(m,id)
  mm = getModel(m,id)
  return MathProgBase.numconstr(mm)
end

function getTotalNumVars(m)
    nvar = 0
    for i=0:num_scenarios(m)
        nvar += getNumVars(m,i)
    end
    return nvar
end

function getTotalNumCons(m)
    ncon = 0
    for i=0:num_scenarios(m)
        ncon += getNumCons(m,i)
    end
    return ncon
end

function getLocalScenarioIDs(m)
  myrank,mysize = getMyRank()
  numScens = num_scenarios(m)
  d = div(numScens,mysize)
  s = myrank * d + 1
  e = myrank == (mysize-1)? numScens:s+d-1
  ids=[0;s:e]
end

function getLocalChildrenIds(m)
    myrank,mysize = getMyRank()
    numScens = num_scenarios(m)
    d = div(numScens,mysize)
    s = myrank * d + 1
    e = myrank == (mysize-1)? numScens:s+d-1
    ids = collect(s:e)
end

end # module StructJuMPSolverInterface


include("pips_parallel.jl")
#include("./solver/pips_serial.jl")
#include("./solver/ipopt_serial.jl")
