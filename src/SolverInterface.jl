try
    include(get(ENV,"PIPS_NLP_JULIA_INTERFACE",""))
catch err
    if(isa(err, ErrorException))
      warn("Could not include PIPS-NLP Julia interface file. Please setup ENV variable 'PIPS_NLP_JULIA_INTERFACE' to the location of this file, usually in PIPS repo at PIPS-NLP/JuliaInterface/ParPipsNlp.jl")
    end
    rethrow()
end

try
    include(get(ENV,"PIPS_NLP_PAR_JULIA_INTERFACE",""))
catch err
    if(isa(err, ErrorException))
      warn("Could not include PIPS-NLP Julia interface file. Please setup ENV variable 'PIPS_NLP_PAR_JULIA_INTERFACE' to the location of this file, usually in PIPS repo at PIPS-NLP/JuliaInterface/ParPipsNlp.jl")
    end
    rethrow()
end

module SolverInterface

# package code goes here
include("helper.jl")
include("structure_helper.jl")
include("nonstruct_helper.jl")
include("./solver/ipopt_interface.jl")
include("./solver/serial_pipsnlp_interface.jl")
include("./solver/structure_pipsnlp_interface.jl")

export SerialIpoptInterface
export SerialPipsNlpInterface
export ParPipsInterface

end # module
