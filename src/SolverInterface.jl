
module SolverInterface


# Struct Model interface
abstract ModelInterface

export ModelInterface


# package code goes here
include("helper.jl")
include("structure_helper.jl")
include("nonstruct_helper.jl")

# include("./solver/ipopt_interface.jl")
# include("./solver/serial_pipsnlp_interface.jl")
# include("./solver/structure_pipsnlp_interface.jl")

# export SerialIpoptInterface
# export SerialPipsNlpInterface
# export ParPipsInterface

end # module
