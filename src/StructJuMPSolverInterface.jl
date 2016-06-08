#
# front-end for StructJuMP solvers interfaces
#

module StructJuMPSolverInterface

import MPI

# Struct Model interface
abstract ModelInterface

export ModelInterface, KnownSolvers, sj_solve, getModel, getVarValue, getVarValues, getNumVars, 
        getNumCons, getTotalNumVars, getTotalNumCons, getLocalScenarioIds, getLocalChildrenIds,
        getObjectiveVal

const KnownSolvers = Dict{AbstractString,Function}();
const SolverStatus = Dict{Int, Symbol}(
  0=>:Optimal,
  1=>:Infesible,
  2=>:Unbounded,
  3=>:IterationExceeded
  )

function sj_solve(model; solver="Unknown", with_prof=false, suppress_warmings=false,kwargs...)
    if !haskey(KnownSolvers,solver)
        Base.warn("Unknow solver: ", solver)
        Base.error("Known solvers are: ", keys(KnownSolvers))
    end
    status = KnownSolvers[solver](model; with_prof=with_prof, suppress_warmings=false,kwargs...)
    
    if !haskey(SolverStatus,status)
      Base.warn("solver can't solve the problem");
      return :Error
    else
      return SolverStatus[status]
    end
    # @show getLocalChildrenIds(m)

    MPI.Finalize()
end



# package code goes here
include("helper.jl")
# include("structure_helper.jl")
# include("nonstruct_helper.jl")

function getModel(m,id)
    return id==0?m:getchildren(m)[id]
end

function getVarValues(m,id)
    mm = getModel(m,id); 
    v = Float64[];
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
function getObjectiveVal(m)
  #MPI is already finalized , therefore returns only objective value at local nodes
  # mid, nprocs = getMyRank()
  # lobj = 0.0
  # x0 = getVarValues(m,0)

  # if mid == 0
  #   e =  get_nlp_evaluator(m,0)
  #   lobj = MathProgBase.eval_f(e,build_x(m,0,x0,x0))
  # end

  # for i in getLocalChildrenIds(m)
  #     x1 = getVarValues(m,i)
  #     lobj += MathProgBase.eval_f(get_nlp_evaluator(m,i),build_x(m,i,x0,x1))
  # end

  # obj = MPI.Reduce(lobj,MPI.SUM,0,getStructure(m).comm)
  lobj =0.0
  x0 = getVarValues(m,0)
  # @show x0
  e = get_nlp_evaluator(m,0)
  lobj = MathProgBase.eval_f(e,build_x(m,0,x0,x0))
  for i in getchildren(m)
    id = i[1]
    e = get_nlp_evaluator(m,id)
    x1 = getVarValues(m,id)
    lobj += MathProgBase.eval_f(e,build_x(m,id,x0,x1))
  end
  return lobj;
end

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

function getLocalScenarioIds(m)
  myrank,mysize = getMyRank()
  numScens = num_scenarios(m)
  d = div(numScens,mysize)
  s = myrank * d + 1
  e = myrank == (mysize-1)? numScens:s+d-1
  ids=[0;s:e]
end

function getLocalChildrenIds(m)
    myrank,mysize = getMyRank()
    # @show myrank, mysize
    numScens = num_scenarios(m)
    d = div(numScens,mysize)
    s = myrank * d + 1
    e = myrank == (mysize-1)? numScens:s+d-1
    ids = collect(s:e)
end

end # module StructJuMPSolverInterface


include("pips_parallel.jl")
include("pips_serial.jl")
include("ipopt_serial.jl")
