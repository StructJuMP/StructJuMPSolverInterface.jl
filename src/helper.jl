#helper.jl
using StructJuMP, JuMP
import MathProgBase

export  get_model,  get_numcons,  get_numvars, get_var_value, get_nlp_evaluator, convert_to_lower,
        array_copy, write_mat_to_file, convert_to_c_idx, g_numvars, g_numcons, getVarValue,
        getScenarioIds, getVarValue, write_x, getSubScenId, @pips_second_stage, message

function write_x(subdir,iter,x)
    @printf("writing x to ./%s/x%d \n",subdir,iter)
    run(`mkdir -p ./$subdir`)
    writedlm(string("./",subdir,"/x",iter),x,",")
end

function message(s)
    rank, nprocs = getMyRank()
    @printf("[%d/%d] [ %s ] \n", rank, nprocs, s)
end

function getScenarioIds(m::JuMP.Model)
    myrank,mysize = getMyRank()
    numScens = num_scenarios(m)
    d = div(numScens,mysize)
    s = myrank * d + 1
    e = myrank == (mysize-1)? numScens:s+d-1
    ids = [0;s:e]
end

function getSubScenId(m::JuMP.Model)
    myrank,mysize = getMyRank()
    numScens = num_scenarios(m)
    d = div(numScens,mysize)
    s = myrank * d + 1
    e = myrank == (mysize-1)? numScens:s+d-1
    ids = collect(s:e)
end


macro pips_second_stage(m,ind,code)
    show(m)
    show(ind)
    show(code)
    return quote
        proc_idx_set = getScenarioIds($(esc(m)))
        for $(esc(ind)) in proc_idx_set
            $(esc(code))
        end
    end
end

function get_model(m,id)
    return id==0?m:getchildren(m)[id]
end

function getVarValue(m)
    ids = getScenarioIds(m)
    for i in ids
        @printf "At node : %d\n" i
        @printf "\t %s \n" get_var_value(m,i)
    end
end

function get_numvars(m,id)
    mm = get_model(m,id)
    nvar = MathProgBase.numvar(mm) - length(getStructure(mm).othermap)
    return nvar
end

function get_numcons(m,id)
    mm = get_model(m,id)
    return MathProgBase.numconstr(mm)
end

function get_var_value(m,id)
    mm = get_model(m,id)
    v = Float64[];
    for i = 1:get_numvars(m,id)
        v = [v;JuMP.getvalue(JuMP.Variable(mm,i))]
    end
    return v
end

function get_nlp_evaluator(m,id)
    # @show id,getScenarioIds(m)
    # @show getProcIdxSet(m)
    # assert(id == 0 || id in getProcIdxSet(m))
    e = JuMP.NLPEvaluator(get_model(m,id))
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


function g_numcons(m)
    ncon = 0
    for i=0:num_scenarios(m)
        ncon += get_numcons(m,i)
    end
    return ncon
end

function g_numvars(m)
    nvar = 0
    for i=0:num_scenarios(m)
        nvar += get_numvars(m,i)
    end
    return nvar
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

function SparseMatrix.sparse(I,J,V, M, N;keepzeros=false)
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

