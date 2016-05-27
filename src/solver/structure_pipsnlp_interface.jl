module ParPipsNlpInterface

using StructJuMP, JuMP
using MPI
using StructJuMPSolverInterface
using PIPS_NLP, ParPipsNlp

import MathProgBase

# function load_x(subdir,iter)
#     @printf("load x from ./%s/x%d \n",subdir,iter)
#     x = readdlm(string("./",subdir,"/x",iter))
#     x0 = x[1:78]
#     x1 = x[79:144]
#     return x0, x1
# end

# function write_x0(subdir,iter,x)
#     @printf("writing x to ./%s/x0_%d \n",subdir,iter)
#     run(`mkdir -p ./$subdir`)
#     writedlm(string("./",subdir,"/x0_",iter),x,",")
# end
# function write_x1(subdir,iter,x)
#     @printf("writing x to ./%s/x1_%d \n",subdir,iter)
#     run(`mkdir -p ./$subdir`)
#     writedlm(string("./",subdir,"/x1_",iter),x,",")
# end



type StructJuMPModel <: ModelInterface
    internalModel::JuMP.Model
    status::Int
    g_iter::Int
    id_con_idx_map::Dict{Int,Pair{Dict{Int,Int},Dict{Int,Int}}}  #eq, ieq, jump->actual used

    t_init_idx::Float64
    t_init_x0::Float64
    t_prob_info::Float64
    t_eval_f::Float64
    t_eval_g::Float64
    t_eval_grad_f::Float64
    t_eval_jac_g::Float64
    t_eval_h::Float64
    t_write_solution::Float64
    
    get_num_scen::Function
    get_sense::Function
    get_status::Function

    get_num_rows::Function
    get_num_cols::Function
    get_num_eq_cons::Function
    get_num_ineq_cons::Function

    set_status::Function
    
    set_num_rows::Function
    set_num_cols::Function
    set_num_eq_cons::Function
    set_num_ineq_cons::Function

    str_init_x0::Function
    str_prob_info::Function
    str_eval_f::Function
    str_eval_g::Function
    str_eval_grad_f::Function
    str_eval_jac_g::Function
    str_eval_h::Function
    str_write_solution::Function



    function StructJuMPModel(model::JuMP.Model, status::Int)
        instance = new(model,status,-1,
            Dict{Int,Pair{Dict{Int,Int},Dict{Int,Int}}}(),
            0,0,0,0,0,0,0,0,0
            )
        tic()
        init_constraints_idx_map(model,instance.id_con_idx_map)
        
        instance.get_num_scen = function()
            return num_scenarios(instance.internalModel)
        end
        instance.get_sense = function()
            return getobjectivesense(instance.internalModel)
        end
        instance.get_status = function()
            return instance.status
        end
        instance.get_num_rows = function(id::Integer)
            return get_numcons(instance.internalModel,id)
        end
        instance.get_num_cols = function(id::Integer)
            return get_numvars(instance.internalModel,id)
        end
        instance.get_num_eq_cons = function(id::Integer)
            return length(instance.id_con_idx_map[id][1])
        end
        instance.get_num_ineq_cons = function(id::Integer)
            return length(instance.id_con_idx_map[id][2])
        end
        instance.set_status = function(s::Integer)
            instance.status = s
        end
        instance.set_num_rows = function(id::Integer,v::Integer) end
        instance.set_num_cols = function(id::Integer,v::Integer) end
        instance.set_num_eq_cons = function(id::Integer,v::Integer) end
        instance.set_num_ineq_cons = function(id::Integer,v::Integer) end


        instance.str_init_x0 = function(id,x0)
            tic()
            assert(id in getScenarioIds(instance.internalModel))
            mm = get_model(instance.internalModel,id)
            nvar = get_numvars(instance.internalModel,id)
            @assert length(x0) == nvar
            
            for i=1:nvar
                x0[i] = getvalue(Variable(mm,i))
                isnan(x0[i])?x0[i]=1.0:nothing
            end
            # @show x0;
            instance.t_init_x0 += toq() 
        end  

        instance.str_prob_info = function(id,mode,clb,cub,rlb,rub)
            tic()
            # @show id
            if mode == :Structure
                nn = get_numvars(instance.internalModel,id)
                mm = get_numcons(instance.internalModel,id)
                # @show nn,mm
                instance.t_prob_info += toq()
                return (nn,mm)
            elseif mode == :Values
                # @show length(clb),length(cub)
                mm = get_model(instance.internalModel,id)
                nvar = get_numvars(instance.internalModel,id)
                @assert length(clb) == nvar 
                @assert length(cub) == nvar
                array_copy(mm.colUpper, 1, cub, 1, nvar)
                array_copy(mm.colLower, 1, clb, 1, nvar)
                lb,ub = JuMP.constraintbounds(mm)
                # @show lb, ub
                (eq_idx, ieq_idx) = instance.id_con_idx_map[id]
                # @show eq_idx,ieq_idx
                @assert length(lb) == length(ub)
                @assert length(eq_idx) + length(ieq_idx) == length(lb)
                num_eq = length(eq_idx)
                for i in eq_idx
                    rlb[i[2]] = lb[i[1]]
                    rub[i[2]] = ub[i[1]]
                end
                for i in ieq_idx
                    rlb[i[2]+num_eq] = lb[i[1]]
                    rub[i[2]+num_eq] = ub[i[1]]
                end
                # @show id, rlb, rub
            else
                @assert false mode
            end
            instance.t_prob_info += toq()
        end 


        instance.str_eval_f = function(id,x0,x1)
            tic()
            # x0, x1 = load_x("pips", instance.g_iter)

            e =  get_nlp_evaluator(instance.internalModel,id)
            instance.t_eval_f += toq()
            obj = MathProgBase.eval_f(e,build_x(instance.internalModel,id,x0,x1))

            # @show "eval_f", id, obj
            # @show x0, x1
            return obj
        end

        instance.str_eval_g = function(id,x0,x1, new_eq_g, new_inq_g)
            tic()
            # x0, x1 = load_x("pips", instance.g_iter)

            e = get_nlp_evaluator(instance.internalModel,id)
            g = Vector{Float64}(get_numcons(instance.internalModel,id))
            MathProgBase.eval_g(e,g,build_x(instance.internalModel,id,x0,x1))
            (eq_idx, ieq_idx) = instance.id_con_idx_map[id]
            @assert length(new_eq_g) == length(eq_idx)
            @assert length(ieq_idx) == length(new_inq_g)
            for i in eq_idx
                new_eq_g[i[2]] = g[i[1]]
                @assert !haskey(ieq_idx,i[1])
            end
            for i in ieq_idx
                @assert !haskey(eq_idx,i[1])
                new_inq_g[i[2]] = g[i[1]]
            end
            instance.t_eval_g  += toq()
            # @show "********  str_eval_g ", id , x0, x1 
            # @show new_eq_g, new_inq_g
        end

        instance.str_eval_grad_f = function(rowid, colid, x0, x1, new_grad_f)
            tic()
            # x0, x1 = load_x("pips", instance.g_iter)

            @assert rowid >= colid
            @assert sum(new_grad_f) == 0.0 
            e = get_nlp_evaluator(instance.internalModel,rowid)
            x = build_x(instance.internalModel,rowid,x0,x1)
            g = Vector{Float64}(length(x))
            MathProgBase.eval_grad_f(e,g,x)
            @assert length(g) == MathProgBase.numvar(get_model(instance.internalModel,rowid))
            @assert length(new_grad_f) == get_numvars(instance.internalModel,colid)

            var_idx_map = get_jac_col_var_idx(instance.internalModel,rowid,colid)
            for i in var_idx_map
                new_grad_f[i[2]] = g[i[1]]
            end
            instance.t_eval_grad_f += toq()

            # @show "**********  str_eval_grad_f", rowid, colid, x0, x1
            # @show new_grad_f
        end

        instance.str_eval_jac_g = function(rowid, colid, x0 , x1, mode, e_rowidx, e_colptr, e_values, i_rowidx, i_colptr, i_values)
            # @show "**********  str_eval_jac_g", mode, rowid, colid, x0, x1
            tic()
            # x0, x1 = load_x("pips",instance.g_iter)

            # @show "str_eval_jac_g", rowid, colid, mode
            m = instance.internalModel
            @assert rowid<=num_scenarios(m) && colid <= num_scenarios(m)
            @assert rowid >= colid
            if(mode == :Structure)
                e = get_nlp_evaluator(m,rowid)
                (jac_I,jac_J) = MathProgBase.jac_structure(e)
                # @show jac_I
                # @show jac_J
                (eq_idx, ieq_idx) = instance.id_con_idx_map[rowid]
                var_idx = get_jac_col_var_idx(m,rowid,colid)
                # @show var_idx

                eq_jac_I = Vector{Int}() 
                ieq_jac_I = Vector{Int}()
                eq_jac_J = Vector{Int}()
                ieq_jac_J = Vector{Int}()
                for i = 1:length(jac_I)
                    ii = jac_I[i]
                    jj = jac_J[i]
                    if haskey(var_idx,jj)
                        if haskey(eq_idx,ii)
                            push!(eq_jac_I, eq_idx[ii])
                            push!(eq_jac_J, var_idx[jj])
                        else
                            @assert haskey(ieq_idx,ii)
                            push!(ieq_jac_I, ieq_idx[ii])
                            push!(ieq_jac_J, var_idx[jj])
                        end
                    end
                end

                # @show eq_jac_I
                # @show eq_jac_J
                eq_jac = sparse(eq_jac_I,eq_jac_J,ones(Float64,length(eq_jac_I)),length(eq_idx),get_numvars(m,colid))
                # @show ieq_jac_I
                # @show ieq_jac_J
                ieq_jac = sparse(ieq_jac_I,ieq_jac_J,ones(Float64,length(ieq_jac_I)),length(ieq_idx),get_numvars(m,colid))
                # @show (length(eq_jac.nzval), length(ieq_jac.nzval))
                instance.t_eval_jac_g += toq() 
                return (length(eq_jac.nzval), length(ieq_jac.nzval))
            elseif(mode == :Values)
                e = get_nlp_evaluator(m,rowid)
                (jac_I,jac_J) = MathProgBase.jac_structure(e)
                jac_g = Vector{Float64}(length(jac_I))
                MathProgBase.eval_jac_g(e,jac_g,build_x(m,rowid,x0,x1))
                (eq_idx, ieq_idx) = instance.id_con_idx_map[rowid]
                var_idx = get_jac_col_var_idx(m,rowid,colid)

                eq_jac_I = Vector{Int}() 
                ieq_jac_I = Vector{Int}()
                eq_jac_J = Vector{Int}()
                ieq_jac_J = Vector{Int}()
                eq_jac_g = Vector{Float64}()
                ieq_jac_g = Vector{Float64}()
                for i = 1:length(jac_I)
                    ii = jac_I[i]
                    jj = jac_J[i]
                    vv = jac_g[i]
                    # @show ii, jj, vv
                    if haskey(var_idx,jj)
                        if haskey(eq_idx,ii)
                            push!(eq_jac_I, eq_idx[ii])
                            push!(eq_jac_J, var_idx[jj])
                            push!(eq_jac_g, vv)
                        else
                            @assert haskey(ieq_idx,ii)
                            push!(ieq_jac_I, ieq_idx[ii])
                            push!(ieq_jac_J, var_idx[jj])
                            push!(ieq_jac_g, vv)
                        end
                    end
                end

                if(length(eq_jac_g) != 0)
                    eq_jac = sparse(eq_jac_I,eq_jac_J,eq_jac_g, length(eq_idx),get_numvars(m,colid), keepzeros=true)
                    # @printf("em=%d; en=%d;\n", length(eq_idx), get_numvars(m,colid))
                    # @show eq_jac_I, eq_jac_J, eq_jac_g
                    
                    array_copy(eq_jac.rowval,1,e_rowidx,1,length(eq_jac.rowval))
                    array_copy(eq_jac.colptr,1,e_colptr,1,length(eq_jac.colptr))
                    array_copy(eq_jac.nzval, 1,e_values,1,length(eq_jac.nzval))
                    convert_to_c_idx(e_rowidx)
                    convert_to_c_idx(e_colptr)

                    filename = string("jaceq_",rowid,"_",colid)
                    write_mat_to_file(filename,eq_jac)
                    # @show eq_jac
                end

                if(length(ieq_jac_g) != 0)
                    ieq_jac = sparse(ieq_jac_I,ieq_jac_J,ieq_jac_g, length(ieq_idx),get_numvars(m,colid), keepzeros=true)
                    # @printf("im=%d; in=%d;\n", length(ieq_idx), get_numvars(m,colid))
                    # @show ieq_jac_I, ieq_jac_J, ieq_jac_g
                    
                    array_copy(ieq_jac.rowval,1,i_rowidx,1,length(ieq_jac.rowval))
                    array_copy(ieq_jac.colptr,1,i_colptr,1,length(ieq_jac.colptr))
                    array_copy(ieq_jac.nzval, 1,i_values,1,length(ieq_jac.nzval))
                    convert_to_c_idx(i_rowidx)
                    convert_to_c_idx(i_colptr)
                    
                    filename = string("jacieq_",rowid,"_",colid)
                    write_mat_to_file(filename,ieq_jac)
                    # @show ieq_jac
                end
            else
                @assert false mode
            end
            instance.t_eval_jac_g += toq() 
            # @show "end str_eval_jac_g"
        end


        instance.str_eval_h = function(rowid, colid, x0, x1, obj_factor, lambda, mode, rowidx, colptr, values)
            tic()
            # x0, x1 = load_x("pips", instance.g_iter)

            # @show "**********  str_eval_h", mode, rowid, colid, obj_factor
            # @show x0, x1
            # @show lambda
            m = instance.internalModel
            @assert rowid<=num_scenarios(m) && colid <=num_scenarios(m)
            if(mode == :Structure)
                if rowid == colid  #diagonal
                    e = get_nlp_evaluator(m,rowid)
                    (h_J,h_I) = MathProgBase.hesslag_structure(e) # upper trangular
                    col_var_idx, row_var_idx = get_h_var_idx(m,rowid,colid)
                    new_h_I = Vector{Int}()
                    new_h_J = Vector{Int}()
                    for i = 1:length(h_I)
                        ii = h_I[i]
                        jj = h_J[i]
                        if haskey(col_var_idx,jj) && haskey(row_var_idx,ii)
                            new_ii = row_var_idx[ii]
                            new_jj = col_var_idx[jj]
                            if new_ii < new_jj
                                push!(new_h_I,new_ii)
                                push!(new_h_J,new_jj)
                            else
                                push!(new_h_I,new_jj)
                                push!(new_h_J,new_ii)
                            end
                        end
                    end
                    laghess = sparse(new_h_I,new_h_J, ones(Float64,length(new_h_I)))
                    instance.t_eval_h += toq()
                    return length(laghess.nzval)
                elseif rowid == 0  && colid != 0 #border corss hessian
                    e = get_nlp_evaluator(m,colid)
                    (h_J,h_I) = MathProgBase.hesslag_structure(e) # upper trangular
                    col_var_idx, row_var_idx = get_h_var_idx(m,rowid,colid)
                    new_h_I = Vector{Int}()
                    new_h_J = Vector{Int}()
                    for i = 1:length(h_I)
                        ii = h_I[i]
                        jj = h_J[i]
                        if haskey(col_var_idx,jj) && haskey(row_var_idx,ii)
                            new_ii = row_var_idx[ii]
                            new_jj = col_var_idx[jj]
                            push!(new_h_I,new_ii)
                            push!(new_h_J,new_jj)
                        end
                    end
                    laghess = sparse(new_h_I,new_h_J, ones(Float64,length(new_h_I)))
                    instance.t_eval_h += toq()

                    return length(laghess.nzval)
                else
                    @assert (rowid !=0 && colid == 0)
                    @assert false
                end
            elseif(mode == :Values)
                if rowid == colid || (rowid !=0 && colid == 0) #diagonal or root contribution
                    e = get_nlp_evaluator(m,rowid)
                    (h_J, h_I) = MathProgBase.hesslag_structure(e)
                    h = Vector{Float64}(length(h_I))
                    x = build_x(m,rowid,x0,x1)
                    (eq_idx, ieq_idx) = instance.id_con_idx_map[rowid]
                    numeq = length(eq_idx)
                    lam_new = Vector{Float64}(length(lambda))
                    for i in eq_idx
                        lam_new[i[1]] = lambda[i[2]]
                    end
                    for i in ieq_idx
                        lam_new[i[1]] = lambda[i[2]+numeq]
                    end

                    MathProgBase.eval_hesslag(e,h,x,obj_factor,lam_new)
                    # @show x0,x1
                    # @show x
                    # @show lambda
                    # @show h_I
                    # @show h_J
                    # @show h
                    col_var_idx,row_var_idx = get_h_var_idx(m,rowid, colid)
                    # @show col_var_idx
                    # @show row_var_idx     
                    new_h_I = Vector{Int}()
                    new_h_J = Vector{Int}()
                    new_h = Vector{Float64}()
                    for i = 1:length(h_I)
                        ii = h_I[i]
                        jj = h_J[i]
                        if haskey(col_var_idx,jj) && haskey(row_var_idx,ii)
                            new_ii = row_var_idx[ii]
                            new_jj = col_var_idx[jj]
                            if new_ii < new_jj
                                push!(new_h_I,new_ii)
                                push!(new_h_J,new_jj)
                            else
                                push!(new_h_I,new_jj)
                                push!(new_h_J,new_ii)
                            end
                            push!(new_h, h[i])
                        end
                    end
                    # @show new_h_I
                    # @show new_h_J
                    # @show new_h
                    if (rowid !=0 && colid == 0) #root contribution
                        (h0_J,h0_I) = MathProgBase.hesslag_structure(get_nlp_evaluator(m,0))
                        # @show h0_I,h0_J
                        str_laghess = sparse([new_h_I;h0_I], [new_h_J;h0_J], [new_h;zeros(Float64,length(h0_I))], get_numvars(m,0),get_numvars(m,0), keepzeros=true)
                        
                        # new_h_I = [new_h_I;h0_I]
                        # new_h_J = [new_h_J;h0_J]
                        # new_h = [new_h;zeros(Float64,length(h0_I))]
                        # @printf("m=%d; n=%d; \n",get_numvars(m,0),get_numvars(m,0))
                        # @show new_h_I, new_h_J, new_h

                        array_copy(str_laghess.rowval,1,rowidx,1,length(str_laghess.rowval))
                        array_copy(str_laghess.colptr,1,colptr,1,length(str_laghess.colptr))
                        array_copy(str_laghess.nzval, 1,values,1,length(str_laghess.nzval)) 
                    else
                        @assert rowid == colid
                        str_laghess = sparse(new_h_I,new_h_J,new_h,get_numvars(m,rowid),get_numvars(m,rowid),keepzeros = true)   
                        
                        # @printf("m=%d;n=%d; \n",get_numvars(m,rowid),get_numvars(m,rowid))
                        # @show new_h_I, new_h_J, new_h                    
                        
                        array_copy(str_laghess.rowval,1,rowidx,1,length(str_laghess.rowval))
                        array_copy(str_laghess.colptr,1,colptr,1,length(str_laghess.colptr))
                        array_copy(str_laghess.nzval, 1,values,1,length(str_laghess.nzval))
                    end
                else  #second stage cross hessian
                    @assert rowid == 0 && colid !=0 
                    e = get_nlp_evaluator(m,colid)
                    (h_J,h_I) = MathProgBase.hesslag_structure(e) # upper trangular
                    h = Vector{Float64}(length(h_I))
                    x = build_x(m,colid,x0,x1)
                    MathProgBase.eval_hesslag(e,h,x,obj_factor,lambda)
                    col_var_idx,row_var_idx = get_h_var_idx(m,rowid, colid)
                    # @show h_I, h_J
                    # @show h
                    
                    new_h_I = Vector{Int}()
                    new_h_J = Vector{Int}()
                    new_h = Vector{Float64}()
                    for i = 1:length(h_I)
                        ii = h_I[i]
                        jj = h_J[i]
                        if haskey(col_var_idx,jj) && haskey(row_var_idx,ii)
                            new_ii = row_var_idx[ii]
                            new_jj = col_var_idx[jj]
                            push!(new_h_I,new_ii)
                            push!(new_h_J,new_jj)
                            push!(new_h,h[i])
                        end
                    end
                    
                    str_laghess = sparse(new_h_I,new_h_J, new_h, get_numvars(m,colid), get_numvars(m,rowid), keepzeros =true)
                    
                    # @printf("m=%d; n=%d; \n", get_numvars(m,colid), get_numvars(m,rowid))
                    # @show new_h_I, new_h_J, new_h

                    # write_x0("parpips",instance.g_iter,x0)
                    # write_x1("parpips",instance.g_iter,x1)
                    instance.g_iter += 1

                    array_copy(str_laghess.rowval,1,rowidx,1,length(str_laghess.rowval))
                    array_copy(str_laghess.colptr,1,colptr,1,length(str_laghess.colptr))
                    array_copy(str_laghess.nzval, 1,values,1,length(str_laghess.nzval))
                end
                # @show rowidx,colptr,values
                filename = string("hess_",rowid,"_",colid)
                write_mat_to_file(filename,str_laghess)

                convert_to_c_idx(rowidx)
                convert_to_c_idx(colptr)
                # @show values
            else
                @assert false mode
            end 
            # @show "end str_eval_h"
            instance.t_eval_h += toq()
        end
        
        instance.str_write_solution = function(id, x, y_eq, y_ieq)
            tic()
            # @show id, x, y_eq, y_ieq
            @assert id in getScenarioIds(instance.internalModel)
            @assert length(x) == instance.get_num_cols(id)
            @assert length(y_eq) == instance.get_num_eq_cons(id)
            @assert length(y_ieq) == instance.get_num_ineq_cons(id)

            #write back the primal to JuMP
            mm = get_model(instance.internalModel,id)
            for i = 1:length(x)
                setvalue(Variable(mm,i), x[i])
            end

            instance.t_write_solution += toq()
        end
        
        instance.t_init_idx += toq()
        return instance
    end
end

function show_time(m::ModelInterface)
    t_model_time = m.t_init_idx +
    m.t_init_x0 +
    m.t_prob_info +
    m.t_eval_f +
    m.t_eval_g+
    m.t_eval_grad_f+
    m.t_eval_jac_g+
    m.t_eval_h +
    m.t_write_solution;
    return t_model_time
end

#######
# Linking with PIPS Julia Structure interface
######

# setsolverhook(model,structJuMPSolve)

function structJuMPSolve(model; suppress_warmings=false,kwargs...)
    # @show "solve"
    t_total = 0.0
    tic()
    MPI.Init()

    comm = MPI.COMM_WORLD
    # @show "[$(MPI.Comm_rank(comm))/$(MPI.Comm_size(comm))] create problem "
    
    prob = ParPipsNlp.createProblemStruct(comm, StructJuMPModel(model,0))
    # @show "end createStructJuMPPipsNlpProblem"

    solver_total = 0.0
    tic()
    status = ParPipsNlp.solveProblemStruct(prob)
    solver_total += toq()

    freeProblemStruct(prob)
    t_total += toq()


    modeling_time = show_time(prob.model)
    solver_time = solver_total - modeling_time

    if(0==MPI.Comm_rank(MPI.COMM_WORLD)) 
      @printf "Total time %.4f (initialization=%.3f modelling=%.3f solver=%.3f) (in sec)\n" t_total prob.model.t_init_idx modeling_time solver_time
    end
    MPI.Finalize()

    return status
end

end

