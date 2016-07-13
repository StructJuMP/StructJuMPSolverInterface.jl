#
# Interface for PIPS-NLP parallel (structured interface)
#
include("pips_parallel_cfunc.jl")

module PipsNlpInterface 

using PipsNlpSolver
using StructJuMP, JuMP
using StructJuMPSolverInterface

import MathProgBase

type MatrixStorage 
    p::Pair{Int,Int}
    nnz::Int
    rowidx::Vector{Int}
    colptr::Vector{Int}
    nzval::Vector{Float64}

    reset::Function

    function MatrixStorage(rid,cid,nnz,ridx,cptr,vals)
        p = Pair{Int,Int}(rid,cid)
        instance = new(p,nnz,ridx,cptr,vals)

        instance.reset = function()
            instance.p = Pair{Int,Int}(-1,-1)
            nnz = -1
            empty!(instance.rowidx)
            empty!(instance.colptr)
            empty!(instance.nzval)
        end

        return instance
    end

    function MatrixStorage()
        return MatrixStorage(-1,-1,-1,[],[],[])
    end
end

type StructJuMPModel <: ModelInterface
    internalModel::JuMP.Model
    status::Int
    n_iter::Int
    iMap::Dict{Int,Pair{Dict{Int,Int},Dict{Int,Int}}}  #eq, ieq, jump->actual used
    jMap::Dict{Pair{Int,Int},Dict{Int,Int}}
    hcMap::Dict{Pair{Int,Int},Dict{Int,Int}}
    hrMap::Dict{Pair{Int,Int},Dict{Int,Int}}
    jeqMat::MatrixStorage
    jieqMat::MatrixStorage
    hPartMat::MatrixStorage
    hBordMat::MatrixStorage
    # prof::Bool
    # t_sj_init::Float64
    # t_sj_init_x0::Float64
    # t_sj_str_prob_info::Float64
    
    # t_sj_eval_f::Float64
    # t_sj_eval_g::Float64
    # t_sj_eval_grad_f::Float64
    # t_sj_eval_jac_g::Float64
    # t_sj_eval_h::Float64
    
    # t_sj_write_solution::Float64
    
    get_num_scen::Function
    get_sense::Function
    get_status::Function

    get_num_rows::Function
    get_num_cols::Function
    get_num_eq_cons::Function
    get_num_ineq_cons::Function
    get_num_eq_lcons::Function
    get_num_ineq_lcons::Function

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



    function StructJuMPModel(model::JuMP.Model)
        instance = new(model,0,-1,
            Dict{Int,Pair{Dict{Int,Int},Dict{Int,Int}}}(),
            Dict{Pair{Int,Int},Dict{Int,Int}}(),
            Dict{Pair{Int,Int},Dict{Int,Int}}(),
            Dict{Pair{Int,Int},Dict{Int,Int}}(),
            MatrixStorage(),
            MatrixStorage(),
            MatrixStorage(),
            MatrixStorage()
            # ,prof,0,0,0,0,0,0,0,0,0
            )
        
        init_constraints_idx_map(model,instance.iMap)
        
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
            return getNumCons(instance.internalModel,id)
        end
        instance.get_num_cols = function(id::Integer)
            return getNumVars(instance.internalModel,id)
        end
        instance.get_num_eq_cons = function(id::Integer)
            return length(instance.iMap[id][1])
        end
        instance.get_num_ineq_cons = function(id::Integer)
            return length(instance.iMap[id][2])
        end
        instance.get_num_eq_lcons = function()
            return 0
        end
        instance.get_num_ineq_lcons = function()
            return 0
        end
        instance.set_status = function(s::Integer)
            instance.status = s
        end
        instance.set_num_rows = function(id::Integer,v::Integer) end
        instance.set_num_cols = function(id::Integer,v::Integer) end
        instance.set_num_eq_cons = function(id::Integer,v::Integer) end
        instance.set_num_ineq_cons = function(id::Integer,v::Integer) end


        instance.str_init_x0 = function(id,x0)
            assert(id in getLocalBlocksIds(instance.internalModel))
            mm = getModel(instance.internalModel,id)
            nvar = getNumVars(instance.internalModel,id)
            @assert length(x0) == nvar
            
            for i=1:nvar
                x0[i] = getvalue(Variable(mm,i))
                isnan(x0[i])?x0[i]=1.0:nothing
            end
            # @show x0;
        end  

        instance.str_prob_info = function(id,flag,mode,clb,cub,rlb,rub)
            # @show id
            if mode == :Structure
                if flag == 0
                    nn = getNumVars(instance.internalModel,id)
                    mm = getNumCons(instance.internalModel,id)
                else 
                    @assert flag == 1
                    nn = getNumVars(instance.internalModel,id)
                    # mm = getNumLinkCons() #TODO: this is not yet supported
                    mm = 0
                end
                # @show nn,mm
                return (nn,mm)
            elseif mode == :Values
                if flag == 0
                    # @show length(clb),length(cub)
                    mm = getModel(instance.internalModel,id)
                    nvar = getNumVars(instance.internalModel,id)
                    @assert length(clb) == nvar 
                    @assert length(cub) == nvar
                    array_copy(mm.colUpper, 1, cub, 1, nvar)
                    array_copy(mm.colLower, 1, clb, 1, nvar)
                    lb,ub = JuMP.constraintbounds(mm)
                    # @show lb, ub
                    (eq_idx, ieq_idx) = instance.iMap[id]
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
                    @assert flag == 1
                    @assert length(rlb) == length(rub) == 0 #TODO: not yet support linking constraints syntax
                end
            else
                @assert false mode
            end
        end 


        instance.str_eval_f = function(id,x0,x1)
            # @printf("#*************** eval_f - %d \n", id)
            # @show x0, x1
            # x0, x1 = load_x("pips", instance.n_iter)

            e =  get_nlp_evaluator(instance.internalModel,id)
            obj = MathProgBase.eval_f(e,build_x(instance.internalModel,id,x0,x1))

            # @show obj
            return obj

        end

        instance.str_eval_g = function(id,x0,x1, new_eq_g, new_inq_g)
            # x0, x1 = load_x("pips", instance.n_iter)

            e = get_nlp_evaluator(instance.internalModel,id)
            g = Vector{Float64}(getNumCons(instance.internalModel,id))
            MathProgBase.eval_g(e,g,build_x(instance.internalModel,id,x0,x1))
            (eq_idx, ieq_idx) = instance.iMap[id]
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
            # @printf("#********  str_eval_g - %d \n", id)
            # @show x0, x1 
            # @show new_eq_g, new_inq_g
        end

        instance.str_eval_grad_f = function(rowid, colid, x0, x1, new_grad_f)
            # x0, x1 = load_x("pips", instance.n_iter)
            m = instance.internalModel
            @assert rowid >= colid
            @assert sum(new_grad_f) == 0.0 
            e = get_nlp_evaluator(m,rowid)
            x = build_x(m,rowid,x0,x1)
            g = Vector{Float64}(length(x))
            MathProgBase.eval_grad_f(e,g,x)
            @assert length(g) == MathProgBase.numvar(getModel(m,rowid))
            @assert length(new_grad_f) == getNumVars(m,colid)

            var_idx_map = get_jac_col_idx_map(m,rowid,colid,instance.jMap)
            for i in var_idx_map
                new_grad_f[i[2]] = g[i[1]]
            end

            # @printf("#**********  str_eval_grad_f, %d %d \n", rowid, colid)
            # @show x0, x1, new_grad_f
            # @printf("grad%d%d=new_grad_f; \n",rowid,colid)
        end

        instance.str_eval_jac_g = function(rowid, colid, x0 , x1, mode, e_rowidx, e_colptr, e_values, i_rowidx, i_colptr, i_values)
            # @printf("#**********  str_eval_jac_g  - %s -  %d %d \n", mode, rowid, colid);
            # x0, x1 = load_x("pips",instance.n_iter)

            # @show "str_eval_jac_g", rowid, colid, mode
            m = instance.internalModel
            # @show instance
            # @show instance.jMap
            jMap = instance.jMap
            jeqMat = instance.jeqMat
            jieqMat = instance.jieqMat
            @assert rowid<=num_scenarios(m) && colid <= num_scenarios(m)
            @assert rowid >= colid
            if(mode == :Structure)
                # p = Pair{Int,Int}(rowid,colid)
                # if p == instance.jeqMat.p
                #     return jeqMat.nnz, jieqMat.nnz
                # end

                e = get_nlp_evaluator(m,rowid)
                (jac_I,jac_J) = MathProgBase.jac_structure(e)
                # @show jac_I
                # @show jac_J
                (eq_idx, ieq_idx) = instance.iMap[rowid]

                var_idx = get_jac_col_idx_map(m,rowid,colid,jMap)
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
                eq_jac = sparse(eq_jac_I,eq_jac_J,ones(Float64,length(eq_jac_I)),length(eq_idx),getNumVars(m,colid))
                # @show ieq_jac_I
                # @show ieq_jac_J
                ieq_jac = sparse(ieq_jac_I,ieq_jac_J,ones(Float64,length(ieq_jac_I)),length(ieq_idx),getNumVars(m,colid))
                nnzEqJac = length(eq_jac.nzval)
                nnzIeqJac = length(ieq_jac.nzval)
                

                # if rowid != colid 
                #     @assert colid == 0
                #     var_idx = get_jac_col_idx_map(m,rowid,rowid,instance.jMap)
                #     eq_jac_I = Vector{Int}() 
                #     ieq_jac_I = Vector{Int}()
                #     eq_jac_J = Vector{Int}()
                #     ieq_jac_J = Vector{Int}()
                #     for i = 1:length(jac_I)
                #         ii = jac_I[i]
                #         jj = jac_J[i]
                #         if haskey(var_idx,jj)
                #             if haskey(eq_idx,ii)
                #                 push!(eq_jac_I, eq_idx[ii])
                #                 push!(eq_jac_J, var_idx[jj])
                #             else
                #                 @assert haskey(ieq_idx,ii)
                #                 push!(ieq_jac_I, ieq_idx[ii])
                #                 push!(ieq_jac_J, var_idx[jj])
                #             end
                #         end
                #     end
                #     eq_jac = sparse(eq_jac_I,eq_jac_J,ones(Float64,length(eq_jac_I)),length(eq_idx),getNumVars(m,colid))
                #     ieq_jac = sparse(ieq_jac_I,ieq_jac_J,ones(Float64,length(ieq_jac_I)),length(ieq_idx),getNumVars(m,colid))
                #     instance.jeqMat = MatrixStorage(rowid,rowid,length(eq_jac.nzval),eq_jac.rowval,eq_jac.colptr,eq_jac.nzval)
                #     instance.jieqMat = MatrixStorage(rowid,rowid,length(ieq_jac.nzval),ieq_jac.rowval,ieq_jac.colptr,ieq_jac.nzval)
                # end
                # @show nnzEqJac, nnzIeqJac
                return nnzEqJac,nnzIeqJac
            elseif(mode == :Values)
                p = Pair{Int,Int}(rowid,colid)
                # @show jeqMat
                # @show jieqMat
                if p == jeqMat.p
                    array_copy(jeqMat.rowidx,1,e_rowidx,1,length(jeqMat.rowidx))
                    array_copy(jeqMat.colptr,1,e_colptr,1,length(jeqMat.colptr))
                    array_copy(jeqMat.nzval, 1,e_values,1,length(jeqMat.nzval))
                    
                    array_copy(jieqMat.rowidx,1,i_rowidx,1,length(jieqMat.rowidx))
                    array_copy(jieqMat.colptr,1,i_colptr,1,length(jieqMat.colptr))
                    array_copy(jieqMat.nzval, 1,i_values,1,length(jieqMat.nzval))
                else
                    e = get_nlp_evaluator(m,rowid)
                    (jac_I,jac_J) = MathProgBase.jac_structure(e)
                    jac_g = Vector{Float64}(length(jac_I))
                    MathProgBase.eval_jac_g(e,jac_g,build_x(m,rowid,x0,x1))
                    (eq_idx, ieq_idx) = instance.iMap[rowid]


                    var_idx = get_jac_col_idx_map(m,rowid,colid,jMap)
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
                        eq_jac = sparse(eq_jac_I,eq_jac_J,eq_jac_g, length(eq_idx),getNumVars(m,colid), keepzeros=true)
                        # @printf("em=%d; en=%d;\n", length(eq_idx), getNumVars(m,colid))
                        # @show eq_jac_I, eq_jac_J, eq_jac_g
                        # @printf("ejac%d%d=sparse(eq_jac_I,eq_jac_J,eq_jac_g,em,en); \n",rowid,colid)
                    
                        array_copy(eq_jac.rowval,1,e_rowidx,1,length(eq_jac.rowval))
                        array_copy(eq_jac.colptr,1,e_colptr,1,length(eq_jac.colptr))
                        array_copy(eq_jac.nzval, 1,e_values,1,length(eq_jac.nzval))
                        # convert_to_c_idx(e_rowidx)
                        # convert_to_c_idx(e_colptr)

                        filename = string("jaceq_",rowid,"_",colid)
                        write_mat_to_file(filename,eq_jac)
                        # @show eq_jac
                    end

                    if(length(ieq_jac_g) != 0)
                        ieq_jac = sparse(ieq_jac_I,ieq_jac_J,ieq_jac_g, length(ieq_idx),getNumVars(m,colid), keepzeros=true)
                        # @printf("im=%d; in=%d;\n", length(ieq_idx), getNumVars(m,colid))
                        # @show ieq_jac_I, ieq_jac_J, ieq_jac_g
                        # @printf("ijac%d%d=sparse(ieq_jac_I,ieq_jac_J,ieq_jac_g,im,in); \n",rowid,colid)
                        # @printf("jac%d%d=vcat(ejac%d%d,ijac%d%d) \n",rowid,colid,rowid,colid,rowid,colid)
                        
                        array_copy(ieq_jac.rowval,1,i_rowidx,1,length(ieq_jac.rowval))
                        array_copy(ieq_jac.colptr,1,i_colptr,1,length(ieq_jac.colptr))
                        array_copy(ieq_jac.nzval, 1,i_values,1,length(ieq_jac.nzval))
                        # convert_to_c_idx(i_rowidx)
                        # convert_to_c_idx(i_colptr)
                        
                        filename = string("jacieq_",rowid,"_",colid)
                        write_mat_to_file(filename,ieq_jac)
                        # @show ieq_jac
                    end

                    if rowid!=colid
                        @assert colid == 0
                        var_idx = get_jac_col_idx_map(m,rowid,rowid,jMap)
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
                            eq_jac = sparse(eq_jac_I,eq_jac_J,eq_jac_g, length(eq_idx),getNumVars(m,rowid), keepzeros=true)
                            # @printf("em=%d; en=%d;\n", length(eq_idx), getNumVars(m,colid))
                            # @show eq_jac_I, eq_jac_J, eq_jac_g
                            # @printf("ejac%d%d=sparse(eq_jac_I,eq_jac_J,eq_jac_g,em,en); \n",rowid,colid)
                            instance.jeqMat = MatrixStorage(rowid,rowid,length(eq_jac.nzval),eq_jac.rowval,eq_jac.colptr,eq_jac.nzval)
                            # @show eq_jac
                        end

                        if(length(ieq_jac_g) != 0)
                            ieq_jac = sparse(ieq_jac_I,ieq_jac_J,ieq_jac_g, length(ieq_idx),getNumVars(m,rowid), keepzeros=true)
                            # @printf("im=%d; in=%d;\n", length(ieq_idx), getNumVars(m,colid))
                            # @show ieq_jac_I, ieq_jac_J, ieq_jac_g
                            # @printf("ijac%d%d=sparse(ieq_jac_I,ieq_jac_J,ieq_jac_g,im,in); \n",rowid,colid)
                            # @printf("jac%d%d=vcat(ejac%d%d,ijac%d%d) \n",rowid,colid,rowid,colid,rowid,colid)
                            instance.jieqMat = MatrixStorage(rowid,rowid,length(ieq_jac.nzval),ieq_jac.rowval,ieq_jac.colptr,ieq_jac.nzval)
                            # @show ieq_jac
                        end
                    end
                end
                convert_to_c_idx(e_rowidx)
                convert_to_c_idx(e_colptr)
                convert_to_c_idx(i_rowidx)
                convert_to_c_idx(i_colptr)
            else
                @assert false mode
            end

            # @show "end str_eval_jac_g"
        end


        instance.str_eval_h = function(rowid, colid, x0, x1, obj_factor, lambda, mode, rowidx, colptr, values)
            # @printf("#**********  str_eval_h - %s - %d %d  %f \n", mode, rowid, colid, obj_factor)
            # @show lambda
            # x0, x1 = load_x("pips", instance.n_iter)

            # @show x0, x1
            # @show lambda
            m = instance.internalModel
            hcMap = instance.hcMap
            hrMap = instance.hrMap
            hPartMat = instance.hPartMat
            hBordMat = instance.hBordMat
            @assert rowid<=num_scenarios(m) && colid <=num_scenarios(m)
            if(mode == :Structure)
                if rowid == colid  #diagonal
                    e = get_nlp_evaluator(m,rowid)
                    (h_J,h_I) = MathProgBase.hesslag_structure(e) # upper trangular
                    col_var_idx, row_var_idx = get_h_col_idx_map(m,rowid,colid,hcMap,hrMap)
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
                    return length(laghess.nzval)
                elseif rowid == 0  && colid != 0 #border corss hessian
                    e = get_nlp_evaluator(m,colid)
                    (h_J,h_I) = MathProgBase.hesslag_structure(e) # upper trangular
                    col_var_idx, row_var_idx = get_h_col_idx_map(m,rowid,colid,hcMap,hrMap)
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
                    
                    return length(laghess.nzval)
                else
                    @assert (rowid !=0 && colid == 0)
                    @assert false
                end
            elseif(mode == :Values)
                p = Pair{Int,Int}(rowid,colid)
                # @show hBordMat
                # @show hPartMat
                if p == hPartMat.p
                    array_copy(hPartMat.rowidx,1,rowidx,1,length(hPartMat.rowidx))
                    array_copy(hPartMat.colptr,1,colptr,1,length(hPartMat.colptr))
                    array_copy(hPartMat.nzval, 1,values,1,length(hPartMat.nzval))
                elseif p == hBordMat.p
                    array_copy(hBordMat.rowidx,1,rowidx,1,length(hBordMat.rowidx))
                    array_copy(hBordMat.colptr,1,colptr,1,length(hBordMat.colptr))
                    array_copy(hBordMat.nzval, 1,values,1,length(hBordMat.nzval))
                else
                    instance.hPartMat = MatrixStorage()
                    instance.hBordMat = MatrixStorage()
                    if rowid == colid == 0 
                        e = get_nlp_evaluator(m,rowid)
                        (h_J, h_I) = MathProgBase.hesslag_structure(e)
                        h = Vector{Float64}(length(h_I))
                        x = build_x(m,rowid,x0,x1)
                        (eq_idx, ieq_idx) = instance.iMap[rowid]
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
                        col_var_idx,row_var_idx = get_h_col_idx_map(m,rowid, colid,hcMap,hrMap)
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
                        str_laghess = sparse(new_h_I,new_h_J,new_h,getNumVars(m,rowid),getNumVars(m,rowid),keepzeros = true)
                        # @printf("m=%d;n=%d; \n",getNumVars(m,rowid),getNumVars(m,rowid))
                        # @show new_h_I, new_h_J, new_h 
                        # @printf(" hess%d%d=sparse(new_h_I,new_h_J,new_h, m, n); \n",rowid,colid)                   
                        # @show str_laghess
                        array_copy(str_laghess.rowval,1,rowidx,1,length(str_laghess.rowval))
                        array_copy(str_laghess.colptr,1,colptr,1,length(str_laghess.colptr))
                        array_copy(str_laghess.nzval, 1,values,1,length(str_laghess.nzval))
                    else
                        e = get_nlp_evaluator(m,rowid)
                        (h_J, h_I) = MathProgBase.hesslag_structure(e)
                        h = Vector{Float64}(length(h_I))
                        x = build_x(m,rowid,x0,x1)
                        (eq_idx, ieq_idx) = instance.iMap[rowid]
                        numeq = length(eq_idx)
                        lam_new = Vector{Float64}(length(lambda))
                        for i in eq_idx
                            lam_new[i[1]] = lambda[i[2]]
                        end
                        for i in ieq_idx
                            lam_new[i[1]] = lambda[i[2]+numeq]
                        end

                        MathProgBase.eval_hesslag(e,h,x,obj_factor,lam_new)

                        #diagonal
                        @assert rowid == colid
                        col_var_idx,row_var_idx = get_h_col_idx_map(m,rowid, rowid,hcMap,hrMap)
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
                        str_laghess = sparse(new_h_I,new_h_J,new_h,getNumVars(m,rowid),getNumVars(m,rowid),keepzeros = true)   
                        
                        # @printf("m=%d;n=%d; \n",getNumVars(m,rowid),getNumVars(m,rowid))
                        # @show new_h_I, new_h_J, new_h 
                        # @printf(" hess%d%d=sparse(new_h_I,new_h_J,new_h, m, n); \n",rowid,colid)                   
                        
                        array_copy(str_laghess.rowval,1,rowidx,1,length(str_laghess.rowval))
                        array_copy(str_laghess.colptr,1,colptr,1,length(str_laghess.colptr))
                        array_copy(str_laghess.nzval, 1,values,1,length(str_laghess.nzval))
                        
                        #linking border
                        # @assert rowid ==0 && colid !=0
                        # e = get_nlp_evaluator(m,colid)
                        # (h_J,h_I) = MathProgBase.hesslag_structure(e) # upper trangular
                        # h = Vector{Float64}(length(h_I))
                        # x = build_x(m,colid,x0,x1)
                        # MathProgBase.eval_hesslag(e,h,x,obj_factor,lambda)
                        col_var_idx,row_var_idx = get_h_col_idx_map(m,0, colid,hcMap,hrMap)
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
                        
                        str_laghess = sparse(new_h_I,new_h_J, new_h, getNumVars(m,rowid), getNumVars(m,0), keepzeros =true)
                        
                        # @printf("m=%d; n=%d; \n", getNumVars(m,colid), getNumVars(m,rowid))
                        # @show new_h_I, new_h_J, new_h
                        # @printf(" hess%d%d=sparse(new_h_I,new_h_J,new_h, m, n); \n",rowid,colid)

                        # write_x0("parpips",instance.n_iter,x0)
                        # write_x1("parpips",instance.n_iter,x1)

                        # array_copy(str_laghess.rowval,1,rowidx,1,length(str_laghess.rowval))
                        # array_copy(str_laghess.colptr,1,colptr,1,length(str_laghess.colptr))
                        # array_copy(str_laghess.nzval, 1,values,1,length(str_laghess.nzval))
                        instance.hBordMat = MatrixStorage(0,colid,length(str_laghess.nzval),str_laghess.rowval,str_laghess.colptr,str_laghess.nzval)

                        #root contribution
                        col_var_idx,row_var_idx = get_h_col_idx_map(m,rowid, 0,hcMap,hrMap)
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
                        (h0_J,h0_I) = MathProgBase.hesslag_structure(get_nlp_evaluator(m,0))
                        # @show h0_I,h0_J
                        str_laghess = sparse([new_h_I;h0_I], [new_h_J;h0_J], [new_h;zeros(Float64,length(h0_I))], getNumVars(m,0),getNumVars(m,0), keepzeros=true)
                        
                        # new_h_I = [new_h_I;h0_I]
                        # new_h_J = [new_h_J;h0_J]
                        # new_h = [new_h;zeros(Float64,length(h0_I))]
                        # @printf("m=%d; n=%d; \n",getNumVars(m,0),getNumVars(m,0))
                        # @show new_h_I, new_h_J, new_h
                        # @printf(" hess%d%d=sparse(new_h_I,new_h_J,new_h, m, n); \n",rowid,colid)

                        # array_copy(str_laghess.rowval,1,rowidx,1,length(str_laghess.rowval))
                        # array_copy(str_laghess.colptr,1,colptr,1,length(str_laghess.colptr))
                        # array_copy(str_laghess.nzval, 1,values,1,length(str_laghess.nzval)) 
                        instance.hPartMat = MatrixStorage(rowid,0,length(str_laghess.nzval),str_laghess.rowval,str_laghess.colptr,str_laghess.nzval)
                    end
                end

                # filename = string("hess_",rowid,"_",colid)
                # write_mat_to_file(filename,str_laghess)

                convert_to_c_idx(rowidx)
                convert_to_c_idx(colptr)
                # @show values
            else
                @assert false mode
            end 
            # @show "end str_eval_h"
        end
        
        instance.str_write_solution = function(id, x, y_eq, y_ieq)
            # @show id, x, y_eq, y_ieq
            @assert id in getLocalBlocksIds(instance.internalModel)
            @assert length(x) == instance.get_num_cols(id)
            @assert length(y_eq) == instance.get_num_eq_cons(id)
            @assert length(y_ieq) == instance.get_num_ineq_cons(id)

            #write back the primal to JuMP
            mm = getModel(instance.internalModel,id)
            for i = 1:length(x)
                setvalue(Variable(mm,i), x[i])
            end
        end

        return instance
    end
end

# function show_time(m::ModelInterface)
#     t_model_evaluation = m.t_sj_init +
#     m.t_sj_init_x0 +
#     m.t_prob_info +
#     m.t_eval_f +
#     m.t_eval_g+
#     m.t_eval_grad_f+
#     m.t_eval_jac_g+
#     m.t_eval_h +
#     m.t_write_solution;
#     return t_model_time
# end

#######
# Linking with PIPS Julia Structure interface
######

function structJuMPSolve(model; with_prof=false, suppress_warmings=false,kwargs...)
    # @show "solve"
    t_sj_lifetime = 0.0
    if with_prof
        tic()
    end
    # MPI.Init() ＃initialize in model loading

    comm = getStructure(model).mpiWrapper.comm
    # @show "[$(MPI.Comm_rank(comm))/$(MPI.Comm_size(comm))] create problem "
    
    t_sj_model_init = 0.0
    if with_prof
        tic()
    end    
    prob = PipsNlpSolver.createProblemStruct(comm, StructJuMPModel(model), with_prof)

    if with_prof
        t_sj_model_init += toq()
    end

    # @show "end createStructJuMPPipsNlpProblem"

    t_sj_solver_total = 0.0
    if with_prof
        tic()
    end
    status = PipsNlpSolver.solveProblemStruct(prob)
    
    if with_prof
        t_sj_solver_total += toq()
    end


    if with_prof
        t_sj_lifetime += toq()
    end


    # solver_time = solver_total - modeling_time
    # if(0==MPI.Comm_rank(MPI.COMM_WORLD)) 
    #   @printf "Total time %.4f (initialization=%.3f modelling=%.3f solver=%.3f) (in sec)\n" t_total prob.model.t_sj_init modeling_time solver_time
    # end
    if with_prof
        mid, nprocs = getMyRank()
        bname = string(split(ARGS[1],"/")[2],"_",num_scenarios(model))
        run(`mkdir -p ./out/$bname`)
        fname = string("./out/",bname,"/",bname,"_",nprocs,".",mid,".jl.txt")
        # @show bname, fname
        tfile = open(fname, "w")
        s1 = @sprintf("[%d/%d] [ t_sj_model_init %f t_sj_solver_total %f  t_sj_lifetime %f ] \n", mid, nprocs, t_sj_model_init, t_sj_solver_total, t_sj_lifetime)
        s2 = @sprintf("[%d/%d] [ t_jl_str_total %f t_jl_eval_total %f  ] \n", mid, nprocs, prob.t_jl_str_total, prob.t_jl_eval_total)
        if(mid == 0)
            @printf("%s",s1)
            @printf("%s",s2)
        end
        @printf(tfile, "%s", s1)
        @printf(tfile, "%s", s2)
        close(tfile)
        n1 = string("./out/",nprocs,".",mid,".c.txt")
        n2 = string("./out/",bname,"/",bname,"_",nprocs,".",mid,".c.txt")
        run(`mv $n1 $n2`)
    end
    return PIPSRetCodeToSolverInterfaceCode[status]
end

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

function init_constraints_idx_map(m,map)
    assert(length(map) == 0)
    for id in getLocalBlocksIds(m)
        e = get_nlp_evaluator(m,id) #initialize the nlp evaluator
        eq_idx = Dict{Int,Int}()
        ieq_idx = Dict{Int,Int}()
        push!(map,id=>Pair(eq_idx,ieq_idx))
        lb,ub=JuMP.constraintbounds(getModel(m,id))

        for i =1:length(lb)
            if lb[i] == ub[i]
                eq_idx[i] = length(eq_idx) + 1
            else
                ieq_idx[i] =length(ieq_idx) + 1 #remember to offset length(eq_idx)
            end
        end
    end
end

function get_jac_col_idx_map(m,rowid,colid,map)
    p = Pair{Int,Int}(rowid,colid)
    if haskey(map,p)
        idx_map = map[p]
    else
        idx_map = get_jac_col_var_idx(m,rowid, colid)
        push!(map,p=>idx_map)
    end
    return idx_map
end

function get_jac_col_var_idx(m,rowid, colid)  #this method customerized for no linking constraint presented
    # @show "get_jac_col_var_idx",rowid,colid
    idx_map = Dict{Int,Int}() #dummy (jump) -> actual used
    if rowid == colid
        nvar = getNumVars(m,rowid)
        for i = 1:nvar
            idx_map[i] = i
        end
    else
        @assert rowid!=0 && rowid != colid
        mm = getModel(m,rowid)
        othermap = getStructure(mm).othermap
        for p in othermap
            pidx = p[1].col
            cidx = p[2].col
            idx_map[cidx] = pidx
        end
    end
    # @show idx_map
    return idx_map
end

function get_h_col_idx_map(m,rowid,colid,cmap,rmap)
    p = Pair{Int,Int}(rowid,colid)
    if haskey(cmap,p)
        cidx_map = cmap[p]
        ridx_map = rmap[p]
    else
        cidx_map,ridx_map = get_h_var_idx(m,rowid,colid)
        push!(cmap,p=>cidx_map)
        push!(rmap,p=>ridx_map)
    end
    return cidx_map, ridx_map
end

function get_h_var_idx(m,rowid, colid)
    # @show "get_h_var_idx",rowid,colid
    col_idx_map = Dict{Int,Int}() #dummy (jump) -> actual used
    row_idx_map = Dict{Int,Int}()
    if rowid == colid
        nvar = getNumVars(m,rowid) #need to place model variable in front of non model variable.
        for i = 1:nvar
            col_idx_map[i] = i
            row_idx_map[i] = i
        end
    elseif rowid == 0  && colid != 0 #border
        mm = getModel(m,colid)
        othermap = getStructure(mm).othermap
        for p in othermap
            pidx = p[1].col
            cidx = p[2].col
            col_idx_map[cidx] = pidx
        end
        for i = 1:getNumVars(m,colid)
            row_idx_map[i] = i
        end
    elseif colid == 0 && rowid != 0 #root contrib.
        mm = getModel(m,rowid)
        othermap = getStructure(mm).othermap
        for p in othermap
            pidx = p[1].col
            cidx = p[2].col
            col_idx_map[cidx] = pidx
            row_idx_map[cidx] = pidx
        end
    else
        @assert false rowid colid
    end
    # @show col_idx_map,row_idx_map
    return col_idx_map,row_idx_map
end


KnownSolvers["PipsNlp"] = PipsNlpInterface.structJuMPSolve

end

