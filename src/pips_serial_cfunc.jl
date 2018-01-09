#
# This module is a wrapper for the C, unstructured (thus serial) interface of PIPS-NLP
#

module PipsNlpSolverSerial

using StructJuMPSolverInterface

export createProblem, solveProblem

try
  sharedLib=ENV["PIPS_NLP_SHARED_LIB"]
  #explicitly check if the file exists (because dlopen sometimes does not throw an error for invalid filenames, resulting in a seg fault)
  if(!isfile(sharedLib))
    error(string("The specified shared library ([", sharedLib, "]) does not exist"))
  end  
  global const libparpipsnlp=Libdl.dlopen(get(ENV,"PIPS_NLP_SHARED_LIB",""))
catch 
  warn("Could not load PIPS-NLP shared library. Make sure the ENV variable 'PIPS_NLP_SHARED_LIB' points to its location, usually in the PIPS repo at PIPS/build_pips/PIPS-NLP/libpipsnlp.so")
  rethrow()
end

function convert_to_c_idx(indicies)
    for i in 1:length(indicies)
        indicies[i] = indicies[i] - 1
    end
end

type PipsNlpProblem
    ref::Ptr{Void}
    n::Int
    m::Int
    x::Vector{Float64}
    g::Vector{Float64}
    obj_val::Float64
    status::Int

    # Callbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h  # Can be nothing
    
    #jac , hess
    nzJac::Int
    nzHess::Int

    # For MathProgBase
    sense::Symbol

    
    function PipsNlpProblem(ref::Ptr{Void}, n, m, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, nzJac, nzHess)
        prob = new(ref, n, m, zeros(Float64, n), zeros(Float64, m), 0.0, 0,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, nzJac, nzHess,
        :Min)
        
        finalizer(prob, freeProblem)
        # Return the object we just made
        prob
    end
end


###########################################################################
# Callback wrappers
###########################################################################
# Objective (eval_f)
function eval_f_wrapper(x_ptr::Ptr{Float64}, obj_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # println(" julia - eval_f_wrapper " ); 
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::PipsNlpProblem
    # Calculate the new objective
    new_obj = convert(Float64, prob.eval_f(unsafe_wrap(Array,x_ptr, prob.n)))::Float64
    # Fill out the pointer
    unsafe_store!(obj_ptr, new_obj)
    # Done
    return Int32(1)
end

# Constraints (eval_g)
function eval_g_wrapper(x_ptr::Ptr{Float64}, g_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # println(" julia - eval_g_wrapper " ); 
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::PipsNlpProblem
    # Calculate the new constraint values
    new_g = unsafe_wrap(Array,g_ptr, prob.m)
    prob.eval_g(unsafe_wrap(Array,x_ptr, prob.n), new_g)
    # Done
    return Int32(1)
end

# Objective gradient (eval_grad_f)
function eval_grad_f_wrapper(x_ptr::Ptr{Float64}, grad_f_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # println(" julia -  eval_grad_f_wrapper " );    
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::PipsNlpProblem
    # Calculate the gradient
    new_grad_f = unsafe_wrap(Array,grad_f_ptr, Int(prob.n))
    prob.eval_grad_f(unsafe_wrap(Array,x_ptr, Int(prob.n)), new_grad_f)
    if prob.sense == :Max
        new_grad_f *= -1.0
    end
    # Done
    return Int32(1)
end

# Jacobian (eval_jac_g)
function eval_jac_g_wrapper(x_ptr::Ptr{Float64}, values_ptr::Ptr{Float64}, iRow::Ptr{Cint}, jCol::Ptr{Cint},  user_data::Ptr{Void})
    # println(" julia -  eval_jac_g_wrapper " );
    # Extract Julia the problem from the pointer  
    #@show user_data  
    prob = unsafe_pointer_to_objref(user_data)::PipsNlpProblem
    #@show prob
    # Determine mode
    mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
    x = unsafe_wrap(Array,x_ptr, prob.n)
    irows = unsafe_wrap(Array,iRow, Int(prob.nzJac))
    kcols = unsafe_wrap(Array,jCol, Int(prob.n+1))
    values = unsafe_wrap(Array,values_ptr, Int(prob.nzJac))
    prob.eval_jac_g(x, mode, irows, kcols, values)
    if mode == :Structure 
	    convert_to_c_idx(irows)
	    convert_to_c_idx(kcols)
	end
    # Done
    return Int32(1)
end

# Hessian
function eval_h_wrapper(x_ptr::Ptr{Float64}, lambda_ptr::Ptr{Float64}, values_ptr::Ptr{Float64}, iRow::Ptr{Cint}, jCol::Ptr{Cint}, user_data::Ptr{Void})
    # println(" julia - eval_h_wrapper " ); 
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::PipsNlpProblem
    # Did the user specify a Hessian
    if prob.eval_h === nothing
        # No Hessian provided
        return Int32(0)
    else
        # Determine mode
        mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
        x = unsafe_wrap(Array,x_ptr, prob.n)
        lambda = unsafe_wrap(Array,lambda_ptr, prob.m)
        irows = unsafe_wrap(Array,iRow, Int(prob.nzHess))
        kcols = unsafe_wrap(Array,jCol, Int(prob.n+1))
        values = unsafe_wrap(Array,values_ptr, Int(prob.nzHess))
        obj_factor = 1.0
        if prob.sense == :Max
            obj_factor *= -1.0
        end
        prob.eval_h(x, mode, irows, kcols, obj_factor, lambda, values)
        if mode == :Structure
	        convert_to_c_idx(irows)
	        convert_to_c_idx(kcols)
        end
        # Done
        return Int32(1)
    end
end

###########################################################################
# C function wrappers
###########################################################################
function createProblem(n::Int,m::Int,
    x_L::Vector{Float64},x_U::Vector{Float64},
    g_L::Vector{Float64},g_U::Vector{Float64},
    nzJac::Int, nzHess::Int,
    eval_f, eval_g,eval_grad_f,eval_jac_g,eval_h)

    @assert n == length(x_L) == length(x_U)
    @assert m == length(g_L) == length(g_U)
    eval_f_cb = cfunction(eval_f_wrapper,Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Void}) )
    eval_g_cb = cfunction(eval_g_wrapper,Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Void}) )
    eval_grad_f_cb = cfunction(eval_grad_f_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Void}) )
    eval_jac_g_cb = cfunction(eval_jac_g_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, Ptr{Void}))
    eval_h_cb = cfunction(eval_h_wrapper, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, Ptr{Void}))
    
    ret = ccall(Libdl.dlsym(libparpipsnlp,:CreatePipsNlpProblem),Ptr{Void},
            (Cint, Cint,
            Ptr{Float64}, Ptr{Float64},
            Ptr{Float64}, Ptr{Float64},
            Cint, Cint,
            Ptr{Void},Ptr{Void},
            Ptr{Void},Ptr{Void}, Ptr{Void}),
            n,m,
            x_L,x_U,
            g_L,g_U,
            nzJac, nzHess,
            eval_f_cb, eval_g_cb,
            eval_grad_f_cb, eval_jac_g_cb, eval_h_cb
            )
    # println(" ccall CreatePipsNlpProblem done ")
    
    if ret == C_NULL
        error("PIPS-NLP: Failed to construct problem.")
    else
        return(PipsNlpProblem(ret, n, m, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, nzJac, nzHess))
    end
end

function solveProblem(prob::PipsNlpProblem)
    # @show "solveProblem"    
    
    final_objval = [0.0]
    ret = ccall(Libdl.dlsym(libparpipsnlp,:PipsNlpSolve), Cint, 
            (Ptr{Void}, Ptr{Float64}, Ptr{Float64}, Any),
            prob.ref, final_objval, prob.x, prob)
    prob.obj_val = final_objval[1]
    prob.status = Int(ret)

    return prob.status
end

function freeProblem(prob::PipsNlpProblem)
    # @show "freeProblem"
    ret = ccall(Libdl.dlsym(libparpipsnlp,:FreePipsNlpProblem),
            Void, (Ptr{Void},),
            prob.ref)
    # @show ret
    return ret
end

end



