
module StructJuMPSolverInterface


# Struct Model interface
abstract ModelInterface

export ModelInterface, KnownSolvers, sj_solve

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

end # module

include("./solver/ipopt_interface.jl")
include("./solver/serial_pipsnlp_interface.jl")
include("./solver/structure_pipsnlp_interface.jl")
