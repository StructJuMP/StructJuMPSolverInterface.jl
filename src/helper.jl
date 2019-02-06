#helper.jl

using StructJuMP, JuMP
import Printf
import SparseArrays
#import MathProgBase

export get_nlp_evaluator, convert_to_lower, array_copy, write_mat_to_file, convert_to_c_idx, write_x
export @declare_second_stage, @timing, @message
export strip_x, build_x
export PIPSRetCodeToSolverInterfaceCode

const PIPSRetCode = Dict{Int, Symbol}(
    0=>:SUCCESSFUL_TERMINATION,
    1=>:NOT_FINISHED,
    2=>:MAX_ITS_EXCEEDED,
    3=>:INFEASIBLE,
    4=>:NEED_FEASIBILITY_RESTORATION,
    5=>:UNKNOWN
    )

const PIPSRetCodeToSolverInterfaceCode = Dict{Int, Int}(
    0=>0,
    1=>8,
    2=>-1,
    3=>2,
    4=>7,
    5=>8
    )

function strip_x(m,id,x,start_idx)
    mm = getModel(m,id)
    nx = getNumVars(m,id)
    new_x = Vector{Float64}(MathProgBase.numvar(mm))
    array_copy(x,start_idx,new_x,1,nx)

    othermap = getStructure(mm).othermap
    for i in othermap
        pid = i[1].col
        cid = i[2].col
        new_x[cid] = x[pid]
        @assert cid > nx
    end
    return new_x
end


function build_x(m,id,x0,x1)
    # @show "build_x", id, length(x0), length(x1)
    # @show x0, x1
    if id==0
        # @show x0
        return x0
    else
        #build x using index tracking info
        mm = getModel(m,id)
        othermap = getStructure(mm).othermap
        new_x = Vector{Float64}(MathProgBase.numvar(mm))
        unsafe_copy!(new_x,1,x1,1,length(x1)) 
        for e in othermap
            pidx = e[1].col
            cidx = e[2].col
            # @show pidx, cidx
            assert(cidx > length(x1))
            new_x[cidx] = x0[pidx]
        end
        # @show new_x
        return new_x
    end
end


function get_nlp_evaluator(m,id)
    # @show id,getScenarioIds(m)
    # @show getProcIdxSet(m)
    # assert(id == 0 || id in getProcIdxSet(m))
    e = JuMP.NLPEvaluator(getModel(m,id))
    MathProgBase.initialize(e,[:Grad,:Jac,:Hess])
    return e
end


function array_copy(src,os, dest, od, n)
    for i=0:n-1
        dest[i+od] = src[i+os]
    end
end

function write_mat_to_file(filename,mat)
    if(false)
        filename = string("./mat/",filename)
        pre_filename = string(filename,"_0")
        i = 0
        while isfile(pre_filename)
            i += 1
            pre_filename = string(filename,"_",i)
        end
        @show "output : ", pre_filename
        writedlm(pre_filename,mat,",")
    end
end

function convert_to_c_idx(indicies)
    for i in 1:length(indicies)
        indicies[i] = indicies[i] - 1
    end
end

function convert_to_lower(I,J,rI,rJ)
    @assert length(I) == length(J) == length(rI) == length(rJ)
    for i in 1:length(I)
        if I[i] < J[i]
            rI = I[i]
            rJ = J[i]
        else
            rI = J[i]
            rJ = I[i]
        end
    end
    return rI, rJ
end

function write_x(subdir,iter,x)
    printf("writing x to ./%s/x%d \n",subdir,iter)
    run(`mkdir -p ./$subdir`)
    writedlm(string("./",subdir,"/x",iter),x,",")
end

macro message(s)
    return quote
        rank, nprocs = getMyRank()
        printf("[%d/%d] [ %s ] \n", rank, nprocs, $(esc(s)))
    end
end

macro declare_second_stage(m,ind,code)
    return quote
        proc_idx_set = getLocalChildrenIds($(esc(m)))
        # @show proc_idx_set
        for $(esc(ind)) in proc_idx_set
            $(esc(code))
        end
    end
end

macro timing(cond,code)
    return quote
        if $(esc(cond))
            $(esc(code))
       end
    end
end

function SparseArrays.sparse(I,J,V, M, N;keepzeros=false)
    if(!keepzeros)
        return sparse(I,J,V,M,N)
    else
        full = sparse(I,J,ones(Float64,length(I)),M,N)
        actual = sparse(I,J,V,M,N)
        fill!(full.nzval,0.0)

        for c = 1:N
            for i=nzrange(actual,c)
                r = actual.rowval[i]
                v = actual.nzval[i]
                if(v!=0)
                    full[r,c] = v
                end
            end  
            # full.nzval[crange] = actual.nzval[crange] 
        end        
        return full
    end
end

