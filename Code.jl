using JuMP, Random, CPLEX, JLD
using PyCall
# include("showGant.jl")

# ---------------------------------- Original Problem --------------------------------------

function MILP(filenm, timeout)
    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    println("J = $J")

    M = load(filenm, "MM")
    println("M = $M")

    p = load(filenm, "pp")
    println("p = $p")

    L = load(filenm, "LL")
    println("L = $L")

    W = load(filenm, "TT")
    println("W = $W")

    E = load(filenm, "SS")
    println("E = $E")

    # Create model
    model = Model(CPLEX.Optimizer)

    # Parameter definition
    Ω = sum(p) # A large number

    # Variables
    @variable(model, X[1:J, 1:L], Bin)  # Which production line or mold table the job is processed on
    @variable(model, Z[1:J, 1:W], Bin)
    @variable(model, Y[1:J, 1:J], Bin)  # Order between two jobs

    @variable(model, C[1:J, 1:M] >= 0) # Completion time of job n at operation s
    @variable(model, Cmax >= 0)    # Completion time

    # Objective function
    @objective(model, Min, Cmax)

    # Constraints
    @constraint(model, [i in 1:J], sum(X[i, l] for l in 1:L) + sum(Z[i, l] for l in 1:W) == 1)
    @constraint(model, [i in 1:J], sum(Z[i, l] for l in 1:W) - E[i] >= 0)
    @constraint(model, [i in 1:J, j in 1:J], Y[i, j] + Y[j, i] <= 1)

    ## Basic relationship between completion time and processing start time 
    @constraint(model, [i in 1:J, m in 2:M], C[i, m] >= C[i, m-1] + p[i, m])
    @constraint(model, [i in 1:J], C[i, 1] >= p[i, 1])

    ## Assembly line
    M_no_3_4 = [1, 2, 5, 6]
    @constraint(model, [i in 1:J-1, j in i+1:J, l in 1:L, m in M_no_3_4], C[i, m] >= C[j, m] + p[i, m] - (3 - Y[i, j] - X[i, l] - X[j, l]) * Ω)
    @constraint(model, [i in 1:J-1, j in i+1:J, l in 1:L, m in M_no_3_4], C[j, m] >= C[i, m] + p[i, m] - (2 + Y[i, j] - X[i, l] - X[j, l]) * Ω)
    ## Shared resource constraint for the third operation SingleServer
    @constraint(model, [i in 1:J-1, j in i+1:J, l1 in 1:L, l2 in 1:L], C[i, 3] >= C[j, 3] + p[i, 3] - (3 - Y[i, j] - X[i, l1] - X[j, l2]) * Ω)
    @constraint(model, [i in 1:J-1, j in i+1:J, l1 in 1:L, l2 in 1:L], C[j, 3] >= C[i, 3] + p[j, 3] - (2 + Y[i, j] - X[i, l1] - X[j, l2]) * Ω)

    ## Constraints on fixed mold tables
    @constraint(model, [i in 1:J-1, j in i+1:J, w in 1:W], C[i, 1] >= C[j, 6] + p[i, 1] - (3 - Y[i, j] - Z[i, w] - Z[j, w]) * Ω)
    @constraint(model, [i in 1:J-1, j in i+1:J, w in 1:W], C[j, 1] >= C[i, 6] + p[j, 1] - (2 + Y[i, j] - Z[i, w] - Z[j, w]) * Ω)
    ## Shared resource constraint for the third operation SingleServer
    @constraint(model, [i in 1:J-1, j in i+1:J, w1 in 1:W, w2 in 1:W], C[i, 3] >= C[j, 3] + p[i, 3] - (3 - Y[i, j] - Z[i, w1] - Z[j, w2]) * Ω) # 
    @constraint(model, [i in 1:J-1, j in i+1:J, w1 in 1:W, w2 in 1:W], C[j, 3] >= C[i, 3] + p[j, 3] - (2 + Y[i, j] - Z[i, w1] - Z[j, w2]) * Ω) # 

    ## Completion time constraint
    @constraint(model, [i in 1:J], Cmax >= C[i, M])

    # Solve optimization problem
    JuMP.set_time_limit_sec(model, timeout)  # Set time limit
    JuMP.optimize!(model)    #Solve model

    # Output results


    println("Maximum completion time: ", value(Cmax))
    # # showGant(add, left)
    println("UB: ", objective_value(model))
    println("LB: ", objective_bound(model))
    println("gap: ", round(relative_gap(model), digits=2))
    println("solve time: ", round(solve_time(model), digits=2))
    println("status: ", primal_status(model))
    println("Tstatus: ", termination_status(model))
    if primal_status(model) == MOI.NO_SOLUTION && termination_status(model) == MOI.TIME_LIMIT
        return -1, -1, -1, round(solve_time(model), digits=2)
    end

    return value(Cmax), objective_bound(model), relative_gap(model), round(solve_time(model), digits=2)
end

function CP(filenm, timeout)

    #-----------------------------------------------------------------------------
    # Python call CPLEX to solve CPModel
    #-----------------------------------------------------------------------------
    py"""
    from docplex.cp.model import *
    import docplex.cp.utils_visu as visu 
    import os
    import numpy as np

    # J : job number
    # M : job's process number
    # p : job's operation time
    def commentServerCp(J, M, L, W, p, E, timeout):
        
        # create CPModel
        mdl = CpoModel()
        # Initialize job's operation time
        task = [[mdl.interval_var(name = "J{}-M{}".format(j,m)) for m in range(M)] for j in range(J)]
        tasks = [[[mdl.interval_var(size = p[j][m], optional=True, name = "J{}-M{}-LorT{}".format(j,m,l)) for l in range(L+W)] for m in range(M)] for j in range(J)]
        # Create sequence of operation for each machine
        seq = [[sequence_var([tasks[j][m][l] for j in range(J)], name='M{}-L{}'.format(m,l)) for l in range(L+W)] for m in range(M)]

        SJ = []
        NJ = []
        for j in range(J):    
            if E[j] == 1:
                SJ.append(j)
            else:
                NJ.append(j)

        # makesure job's operation is processed in the same line or table
        for j in range(J):
            for m in range(1,M):
                for l in range(L+W):
                    mdl.add(mdl.presence_of(tasks[j][0][l]) == mdl.presence_of(tasks[j][m][l]))

        M_no_3_4 = [0, 1, 4, 5] 

        for m in M_no_3_4:
            for l in range(L):
                mdl.add(mdl.no_overlap(tasks[j][m][l] for j in range(J)))

        # Production line Shared resource constraints
        mdl.add(mdl.no_overlap(tasks[j][2][l] for j in range(J) for l in range(L)))
        
        # Constraint of jobs processed on fixed mold tables
        for w in range(L, L+W):
            mdl.add(mdl.no_overlap(tasks[j][m][w] for j in range(J) for m in range(M)))
        
        mdl.add(mdl.no_overlap(tasks[j][2][w] for j in range(J) for w in range(L,L+W)))  

        # mdl.add(mdl.no_overlap(tasks[j][2][w] for j in range(J) for w in range(L,L+W)))  

        # the next process will start after this process be executed immediately
        for j in range(J):
            for m in range(1,M):
                for l in range(L+W):
                    mdl.add(mdl.end_before_start(tasks[j][m-1][l], tasks[j][m][l]))

        for j in range(J):
            for m in range(1,M):
                for w in range(L,L+W):
                    mdl.add(mdl.end_at_start(tasks[j][m-1][w], tasks[j][m][w]))

        
        # # chose which line executes the job's operation
        for j in SJ:
            for m in range(M):
                mdl.add(mdl.alternative(task[j][m], [tasks[j][m][l] for l in range(L,L+W)]))

        for j in NJ:
            for m in range(M):
                mdl.add(mdl.alternative(task[j][m], [tasks[j][m][l] for l in range(L+W)]))

        # Force sequences to be all identical on all machines
        for l in range(L+W):
            for m in range(1,M):
                mdl.add(same_sequence(seq[0][l], seq[m][l]))


        mdl.add(mdl.minimize(mdl.max(mdl.end_of(task[j][M-1]) for j in range(J))))


        print('Solving model...')
        res = mdl.solve(TimeLimit=timeout)
        print('Solution: ')
        res.print_solution()

        # Remaining parameters
        opt_value = res.get_objective_values()
        obj_bound = res.get_objective_bounds()
        obj_gap = res.get_objective_gaps()
        total_solve_time = res.get_solve_time()
        # interval_sol = msol.get_value()
        # print(" the interval_sol is ", interval_sol)

        print("type of opt_value in Python is", opt_value)
        
        return opt_value, obj_bound, obj_gap, total_solve_time
    """

    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    println("J = $J")

    M = load(filenm, "MM")
    println("M = $M")

    p = load(filenm, "pp")
    println("p = $p")

    L = load(filenm, "LL")
    println("L = $L")

    W = load(filenm, "TT")
    println("W = $W")

    E = load(filenm, "SS")
    println("E = $E")

    # Call Python function
    best_obj_value, lower_bound, gap, cpu_time = py"commentServerCp"(J, M, L, W, p, E, timeout)

    # best_sol and gap are with type of tuple, so we transform tuple to Number
    best_obj_value_number = best_obj_value[1]
    lower_bound_number = lower_bound[1]
    gap_number = gap[1]
    println("best_obj_value = ", best_obj_value_number)
    println("lower_bound = $lower_bound_number")
    println("gap = ", gap_number)
    println("cpu_time = ", cpu_time)
    # println("type of best_sol is ", typeof(best_sol))
    return best_obj_value_number, lower_bound_number, gap_number, cpu_time
end

# ---------------------------------- OASSP1 --------------------------------------
function CP_OASS1(filenm, timeLimit)

    #-----------------------------------------------------------------------------
    # Create Model
    #-----------------------------------------------------------------------------
    py"""
    from docplex.cp.model import *
    import docplex.cp.utils_visu as visu 
    import os
    import numpy as np

    def line_CP(J, M, L, p, timeLimit):
        mdl = CpoModel()

        tasks = [[[mdl.interval_var(size = p[j][m], optional=True, name = "J{}-M{}-L{}".format(j,m,l)) for l in range(L)] for m in range(M)] for j in range(J)]
        task = [[mdl.interval_var(size = p[j][m], name = "J{}-M{}".format(j,m)) for m in range(M)] for j in range(J)]
        
        sequences = [[sequence_var([tasks[j][m][l] for j in range(J)], name='M{}-L{}'.format(m,l)) for l in range(L)] for m in range(M)]
        
        # for j in range(J):    
        #     for m in range(M): 
        #         mdl.add(mdl.sum([mdl.presence_of(tasks[j][m][l]) for l in range(L)]) == 1)
        
        for j in range(J):
            for m in range(1,M):
                for l in range(L):
                    mdl.add(mdl.presence_of(tasks[j][m-1][l]) == mdl.presence_of(tasks[j][m][l]))

        # # chose which factory executes the job's operation
        for j in range(J):
            for m in range(M):
                mdl.add(mdl.alternative(task[j][m], [tasks[j][m][l] for l in range(L)]))
        
        M_no_3_4 = [0, 1, 2, 4, 5]
        
        # Production line
        # for m in M_no_3_4:
        #     mdl.add(mdl.no_overlap(task[j][m] for j in range(J)))
        for l in range(L):
            for m in M_no_3_4:
                mdl.add(mdl.no_overlap(tasks[j][m][l] for j in range(J)))

        mdl.add(mdl.no_overlap(task[j][2] for j in range(J)))
        
        
        # the next process will start after this process be executed immediately
        for j in range(J):
            for m in range(1,M):
                for l in range(L):
                    mdl.add(mdl.end_before_start(task[j][m-1], task[j][m]))
        
        # Force sequences to be all identical on all machines
        for l in range(L):
            for m in range(1,M):
                mdl.add(same_sequence(sequences[m-1][l], sequences[m][l]))
        
        
        mdl.add(mdl.minimize(mdl.max(mdl.end_of(task[j][M-1]) for j in range(J))))
        
        print('Solving model...')
        res = mdl.solve(TimeLimit=timeLimit)
        print('Solution: ')
        res.print_solution()
        
        # Remaining parameters
        opt_value = res.get_objective_values()
        obj_bound = res.get_objective_bounds()
        obj_gap = res.get_objective_gaps()
        total_solve_time = res.get_solve_time()
        # interval_sol = msol.get_value()
        # print(" the interval_sol is ", interval_sol)
        
        print("type of opt_value in Python is", opt_value)
        
        return opt_value, obj_bound, obj_gap, total_solve_time
    """



    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------


    J = load(filenm, "JJ")
    M = load(filenm, "MM")
    L = load(filenm, "LL")
    p = load(filenm, "pp")

    # timeLimit = 600

    opt_val, obj_bound, obj_gap, solve_time = py"line_CP"(J, M, L, p, timeLimit)
    return opt_val[1], obj_bound[1], obj_gap[1], solve_time
end

function MILP_OASS1(filenm, timeLimit)
    J = load(filenm, "JJ")
    M = load(filenm, "MM")
    L = load(filenm, "LL")
    p = load(filenm, "pp")


    # Create model
    model = Model(CPLEX.Optimizer)

    # Parameter definition
    Ω = sum(p) # A large number

    # Variables
    @variable(model, X[1:J, 1:L], Bin)  # Which production line or mold table the job is processed on
    @variable(model, Y[1:J, 1:J], Bin)  # Order between two jobs

    @variable(model, C[1:J, 1:M] >= 0) # Completion time of job n at operation s
    @variable(model, Cmax >= 0)    # Completion time

    # Objective function
    @objective(model, Min, Cmax)

    # Constraints
    @constraint(model, [i in 1:J], sum(X[i, l] for l in 1:L) == 1)
    @constraint(model, [i in 1:J, j in 1:J], Y[i, j] + Y[j, i] <= 1)

    ## Basic relationship between completion time and processing start time 
    @constraint(model, [i in 1:J, m in 2:M], C[i, m] >= C[i, m-1] + p[i, m])
    @constraint(model, [i in 1:J], C[i, 1] >= p[i, 1])

    ## Assembly line
    M_no_3_4 = [1, 2, 5, 6]
    @constraint(model, [i in 1:J-1, j in i+1:J, l in 1:L, m in M_no_3_4], C[i, m] >= C[j, m] + p[i, m] - (3 - Y[i, j] - X[i, l] - X[j, l]) * Ω)
    @constraint(model, [i in 1:J-1, j in i+1:J, l in 1:L, m in M_no_3_4], C[j, m] >= C[i, m] + p[i, m] - (2 + Y[i, j] - X[i, l] - X[j, l]) * Ω)
    ## Shared resource constraint for the third operation SingleServer
    @constraint(model, [i in 1:J-1, j in i+1:J, l1 in 1:L, l2 in 1:L], C[i, 3] >= C[j, 3] + p[i, 3] - (3 - Y[i, j] - X[i, l1] - X[j, l2]) * Ω)
    @constraint(model, [i in 1:J-1, j in i+1:J, l1 in 1:L, l2 in 1:L], C[j, 3] >= C[i, 3] + p[j, 3] - (2 + Y[i, j] - X[i, l1] - X[j, l2]) * Ω)


    ## Completion time constraint
    @constraint(model, [i in 1:J], Cmax >= C[i, M])

    # Solve optimization problem
    JuMP.set_time_limit_sec(model, timeLimit)  # Set time limit
    JuMP.optimize!(model)    #Solve model

    return JuMP.objective_value(model), JuMP.objective_bound(model), JuMP.relative_gap(model), JuMP.solve_time(model)
end

# ---------------------------------- OASSP2 --------------------------------------
function CP_OASS2(filenm, timeLimit)

    #-----------------------------------------------------------------------------
    # Create Model
    #-----------------------------------------------------------------------------
    py"""
    from docplex.cp.model import *
    import docplex.cp.utils_visu as visu 
    import os
    import numpy as np

    def table_CP(J, M, W, p, timeLimit):
        mdl = CpoModel()

        tasks = [[[mdl.interval_var(size = p[j][m], optional = True,  name = "J{}-M{}-L{}".format(j,m,w)) for w in range(W)] for m in range(M)] for j in range(J)]
        task = [[mdl.interval_var(name = "J{}-M{}".format(j,m)) for m in range(M)] for j in range(J)]
        
        sequences = [[sequence_var([tasks[j][m][w] for j in range(J)], name='M{}-L{}'.format(m,w)) for w in range(W)] for m in range(M)]
        # sequences = [sequence_var([tasks[j][m][w] for j in range(J) for m in range(M)], name='W{}'.format(w)) for w in range(W)]

        # Force sequences to be all identical on all machines
        for w in range(W):
            for m in range(1,M):
                mdl.add(same_sequence(sequences[m-1][w], sequences[m][w]))
        
        for j in range(J):
            for m in range(1,M):
                for w in range(W):
                    mdl.add(mdl.presence_of(tasks[j][0][w]) == mdl.presence_of(tasks[j][m][w]))
        
        M_no_3 = [0, 1, 3, 4, 5]
        # Fixed mold tables
        for w in range(W):
            mdl.add(mdl.no_overlap(tasks[j][m][w] for j in range(J) for m in range(M)))
        
        # mdl.add(mdl.no_overlap(tasks[j][2][w] for j in range(J) for w in range(W)))  
        # # the next process will start after this process be executed immediately
        # for j in range(J):
        #     for m in range(1,M):
        #         for w in range(W):
        #             mdl.add(mdl.end_before_start(tasks[j][m-1][w], tasks[j][m][w]))
        #             mdl.add(mdl.end_at_start(tasks[j][m-1][w], tasks[j][m][w]))


        # # the next process will start after this process be executed immediately
        for j in range(J):
            for m in range(1,M):
                mdl.add(mdl.end_before_start(task[j][m-1], task[j][m]))
                mdl.add(mdl.end_at_start(task[j][m-1], task[j][m]))
        

        mdl.add(mdl.no_overlap(task[j][2] for j in range(J)))  


        # # chose which factory executes the job's operation
        for j in range(J):
            for m in range(M):
                mdl.add(mdl.alternative(task[j][m], [tasks[j][m][w] for w in range(W)]))
        
        # for j in range(J):
        #     for m in range(1,M):
        #         mdl.add(mdl.end_at_start(task[j][m-1], task[j][m]))

        
        mdl.add(mdl.minimize(mdl.max(mdl.end_of(task[j][M-1]) for j in range(J))))
        
        print('Solving model...')
        res = mdl.solve(TimeLimit=timeLimit)
        print('Solution: ')
        res.print_solution()
        
        # Remaining parameters
        opt_value = res.get_objective_values()
        obj_bound = res.get_objective_bounds()
        obj_gap = res.get_objective_gaps()
        total_solve_time = res.get_solve_time()
        # interval_sol = msol.get_value()
        # print(" the interval_sol is ", interval_sol)

        print("type of opt_value in Python is", opt_value)
        
        return opt_value, obj_bound, obj_gap, total_solve_time

    """


    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    M = load(filenm, "MM")
    p = load(filenm, "pp")
    W = load(filenm, "TT")
    # Call Python function
    best_obj_value, lower_bound, gap, cpu_time = py"table_CP"(J, M, W, p, timeLimit)

    # best_sol and gap are with type of tuple, so we transform tuple to Number
    best_obj_value_number = best_obj_value[1]
    lower_bound_number = lower_bound[1]
    gap_number = gap[1]
    println("best_obj_value = ", best_obj_value_number)
    println("lower_bound = $lower_bound_number")
    println("gap = ", gap_number)
    println("cpu_time = ", cpu_time)
    # println("type of best_sol is ", typeof(best_sol))
    return best_obj_value_number, lower_bound_number, gap_number, cpu_time
end

function CP_OASS2_pulse(filenm, timeLimit)
    #-----------------------------------------------------------------------------
    # Create Model
    #-----------------------------------------------------------------------------
    py"""
    from docplex.cp.model import *
    import docplex.cp.utils_visu as visu 
    import os
    import numpy as np

    def table_CP(J, M, W, p, timeLimit):
        mdl = CpoModel()

        tasks = [[[mdl.interval_var(size = p[j][m], optional = True,  name = "J{}-M{}-L{}".format(j,m,w)) for w in range(W)] for m in range(M)] for j in range(J)]
        task = [[mdl.interval_var(size = p[j][m], name = "J{}-M{}".format(j,m)) for m in range(M)] for j in range(J)]
        
        for j in range(J):
            for m in range(1,M):
                for w in range(W):
                    mdl.add(mdl.presence_of(tasks[j][0][w]) == mdl.presence_of(tasks[j][m][w]))
        
        M_no_3 = [0, 1, 3, 4, 5]
        # Fixed mold tables
        for w in range(W):
            mdl.add(mdl.no_overlap(tasks[j][m][w] for j in range(J) for m in range(M)))

        # the next process will start after this process be executed immediately
        # for j in range(J):
        #     for m in range(1,M):
        #         for w in range(W):
        #             mdl.add(mdl.end_before_start(tasks[j][m-1][w], tasks[j][m][w]))
        #             mdl.add(mdl.end_at_start(tasks[j][m-1][w], tasks[j][m][w]))

        # resourceUse = [mdl.pulse(tasks[j][2][w], 1) for j in range(J) for w in range(W)]
        # mdl.add(mdl.sum(resourceUse) <= 1)   

        resourceUse = [mdl.pulse(task[j][2], 1) for j in range(J)]
        mdl.add(mdl.sum(resourceUse) <= 1)    
        for j in range(J):
            for m in range(1,M):
                mdl.add(mdl.end_before_start(task[j][m-1], task[j][m]))
                mdl.add(mdl.end_at_start(task[j][m-1], task[j][m]))

        
        # # chose which factory executes the job's operation
        for j in range(J):
            for m in range(M):
                mdl.add(mdl.alternative(task[j][m], [tasks[j][m][w] for w in range(W)]))
        


        
        mdl.add(mdl.minimize(mdl.max(mdl.end_of(task[j][M-1]) for j in range(J))))
        
        print('Solving model...')
        res = mdl.solve(TimeLimit=timeLimit)
        print('Solution: ')
        res.print_solution()
        
        # Remaining parameters
        opt_value = res.get_objective_values()
        obj_bound = res.get_objective_bounds()
        obj_gap = res.get_objective_gaps()
        total_solve_time = res.get_solve_time()
        # interval_sol = msol.get_value()
        # print(" the interval_sol is ", interval_sol)

        print("type of opt_value in Python is", opt_value)



        return opt_value, obj_bound, obj_gap, total_solve_time


    """


    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    M = load(filenm, "MM")
    p = load(filenm, "pp")
    W = load(filenm, "TT")
    # Call Python function
    best_obj_value, lower_bound, gap, cpu_time = py"table_CP"(J, M, W, p, timeLimit)

    # best_sol and gap are with type of tuple, so we transform tuple to Number
    best_obj_value_number = best_obj_value[1]
    lower_bound_number = lower_bound[1]
    gap_number = gap[1]
    println("best_obj_value = ", best_obj_value_number)
    println("lower_bound = $lower_bound_number")
    println("gap = ", gap_number)
    println("cpu_time = ", cpu_time)
    # println("type of best_sol is ", typeof(best_sol))


    return best_obj_value_number, lower_bound_number, gap_number, cpu_time
end

function MILP_OASS2(filenm, timeLimit)
    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    M = load(filenm, "MM")
    p = load(filenm, "pp")
    W = load(filenm, "TT")

    # Create model
    model = Model(CPLEX.Optimizer)

    # Parameter definition
    Ω = sum(p) # A large number

    # Variables
    @variable(model, X[1:J, 1:W], Bin)  # Which production line or mold table the job is processed on
    @variable(model, Y[1:J, 1:J], Bin)  # Order between two jobs

    @variable(model, C[1:J, 1:M] >= 0) # Completion time of job n at operation s
    @variable(model, Cmax >= 0)    # Completion time

    # Objective function
    @objective(model, Min, Cmax)

    # Constraints
    @constraint(model, [i in 1:J], sum(X[i, w] for w in 1:W) == 1)
    @constraint(model, [i in 1:J, j in 1:J], Y[i, j] + Y[j, i] >= 1)

    ## Basic relationship between completion time and processing start time 
    @constraint(model, [i in 1:J, m in 2:M], C[i, m] >= C[i, m-1] + p[i, m])
    @constraint(model, [i in 1:J], C[i, 1] >= p[i, 1])

    ## Constraints on fixed mold tables
    @constraint(model, [i in 1:J-1, j in i+1:J, w in 1:W], C[i, 1] >= C[j, 6] + p[i, 1] - (3 - Y[i, j] - X[i, w] - X[j, w]) * Ω)
    @constraint(model, [i in 1:J-1, j in i+1:J, w in 1:W], C[j, 1] >= C[i, 6] + p[j, 1] - (2 + Y[i, j] - X[i, w] - X[j, w]) * Ω)
    ## Shared resource constraint for the third operation SingleServer
    @constraint(model, [i in 1:J-1, j in i+1:J], C[i, 3] >= C[j, 3] + p[i, 3] - (1 - Y[i, j]) * Ω) # 
    @constraint(model, [i in 1:J-1, j in i+1:J], C[j, 3] >= C[i, 3] + p[j, 3] - Y[i, j] * Ω) # 

    ## Completion time constraint
    @constraint(model, [i in 1:J], Cmax >= C[i, M])

    # Solve optimization problem
    JuMP.set_time_limit_sec(model, timeLimit)  # Set time limit
    JuMP.optimize!(model)    #Solve model
    println("UB: ", objective_value(model))
    println("LB: ", objective_bound(model))
    println("gap: ", round(relative_gap(model), digits=2))
    println("solve time: ", round(solve_time(model), digits=2))
    return value(Cmax), objective_bound(model), round(relative_gap(model), digits=2), round(solve_time(model), digits=2)
end

function MILP_POS_OASS2(filenm, timeLimit)
    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    M = load(filenm, "MM")
    p = load(filenm, "pp")
    W = load(filenm, "TT")

    # Create model
    model = Model(CPLEX.Optimizer)

    # Parameter definition
    Ω = sum(p) # A large number

    # Variables
    @variable(model, X[1:J, 1:W], Bin)  # Which production line or mold table the job is processed on
    @variable(model, Y[1:J, 1:J], Bin)  # Processing position of the job

    @variable(model, C[1:J, 1:M] >= 0) # Completion time of the job at position i in operation m
    @variable(model, Cmax >= 0)    # Completion time

    # Objective function
    @objective(model, Min, Cmax)

    # Constraints
    @constraint(model, [i in 1:J], sum(X[i, w] for w in 1:W) == 1)
    @constraint(model, [i in 1:J], sum(Y[i, k] for k in 1:J) == 1)
    @constraint(model, [k in 1:J], sum(Y[i, k] for i in 1:J) == 1)

    ## Basic relationship between completion time and processing start time 
    @constraint(model, [k in 1:J, m in 2:M], C[k, m] >= C[k, m-1] + sum(p[i, m] * Y[i, k] for i in 1:J))
    @constraint(model, [k in 1:J], C[k, 1] >= sum(p[i, 1] * Y[i, k] for i in 1:J))

    ## Constraints on fixed mold tables
    @constraint(model, [i in 1:J-1, j in i+1:J, k1 in 1:J-1, k2 in k1+1:J, w in 1:W], C[k2, 1] >= C[k1, 6] + p[j, 1] - (4 - Y[i, k1] - Y[j, k2] - X[i, w] - X[j, w]) * Ω)
    @constraint(model, [j in 1:J-1, i in j+1:J, k1 in 1:J-1, k2 in k1+1:J, w in 1:W], C[k2, 1] >= C[k1, 6] + p[j, 1] - (4 - Y[i, k1] - Y[j, k2] - X[i, w] - X[j, w]) * Ω)
    ## Shared resource constraints
    @constraint(model, [k in 2:J], C[k, 3] >= C[k-1, 3] + sum(p[i, 3] * Y[i, k] for i in 1:J))


    ## Completion time constraint
    @constraint(model, [k in 1:J], Cmax >= C[k, M])

    # Solve optimization problem
    JuMP.set_time_limit_sec(model, timeLimit)  # Set time limit
    JuMP.optimize!(model)    #Solve model
    println("UB: ", objective_value(model))
    println("LB: ", objective_bound(model))
    println("gap: ", round(relative_gap(model), digits=2))
    println("solve time: ", round(solve_time(model), digits=2))
    return value(Cmax), objective_bound(model), round(relative_gap(model), digits=2), round(solve_time(model), digits=2)
end

# ---------------------------------- LBBD --------------------------------------

function subproblem_solve_Unit(filenm, timeLimit, X)

    #-----------------------------------------------------------------------------
    # Create Model
    #-----------------------------------------------------------------------------
    py"""
    from docplex.cp.model import *
    import docplex.cp.utils_visu as visu 
    import os
    import numpy as np

    def table_CP(J, M, W, p, timeLimit):
        mdl = CpoModel()

        tasks = [[[mdl.interval_var(size = p[j][m], optional = True,  name = "J{}-M{}-L{}".format(j,m,w)) for w in range(W)] for m in range(M)] for j in range(J)]
        task = [[mdl.interval_var(size = p[j][m], name = "J{}-M{}".format(j,m)) for m in range(M)] for j in range(J)]
        
        sequences = [[sequence_var([tasks[j][m][w] for j in range(J)], name='M{}-L{}'.format(m,w)) for w in range(W)] for m in range(M)]
        # sequences = [sequence_var([tasks[j][m][w] for j in range(J) for m in range(M)], name='W{}'.format(w)) for w in range(W)]

        # Force sequences to be all identical on all machines
        for w in range(W):
            for m in range(1,M):
                mdl.add(same_sequence(sequences[m-1][w], sequences[m][w]))
        
        for j in range(J):
            for m in range(1,M):
                for w in range(W):
                    mdl.add(mdl.presence_of(tasks[j][0][w]) == mdl.presence_of(tasks[j][m][w]))
        
        M_no_3 = [0, 1, 3, 4, 5]
        # Fixed mold tables
        for w in range(W):
            mdl.add(mdl.no_overlap(tasks[j][m][w] for j in range(J) for m in range(M)))
        
        # mdl.add(mdl.no_overlap(tasks[j][2][w] for j in range(J) for w in range(W)))  
        # # the next process will start after this process be executed immediately
        # for j in range(J):
        #     for m in range(1,M):
        #         for w in range(W):
        #             mdl.add(mdl.end_before_start(tasks[j][m-1][w], tasks[j][m][w]))
        #             mdl.add(mdl.end_at_start(tasks[j][m-1][w], tasks[j][m][w]))
        
        # the next process will start after this process be executed immediately
        for j in range(J):
            for m in range(1,M):
                    mdl.add(mdl.end_before_start(task[j][m-1], task[j][m]))
                    mdl.add(mdl.end_at_start(task[j][m-1], task[j][m]))

        resourceUse = [mdl.pulse(task[j][2], 1) for j in range(J)]
        mdl.add(mdl.sum(resourceUse) <= 1)   

        # # chose which factory executes the job's operation
        for j in range(J):
            for m in range(M):
                mdl.add(mdl.alternative(task[j][m], [tasks[j][m][w] for w in range(W)]))
        
        mdl.add(mdl.minimize(mdl.max(mdl.end_of(task[j][M-1]) for j in range(J))))
        
        print('Solving model...')
        res = mdl.solve(TimeLimit=timeLimit)
        print('Solution: ')
        res.print_solution()
        
        # Remaining parameters
        opt_value = res.get_objective_values()
        obj_bound = res.get_objective_bounds()
        obj_gap = res.get_objective_gaps()
        total_solve_time = res.get_solve_time()
        # interval_sol = msol.get_value()
        # print(" the interval_sol is ", interval_sol)
        
        print("type of opt_value in Python is", opt_value)
        
        return opt_value, total_solve_time, obj_bound



    def line_CP(J, M, L, p, timeLimit):
        mdl = CpoModel()

        tasks = [[[mdl.interval_var(size = p[j][m], optional=True, name = "J{}-M{}-L{}".format(j,m,l)) for l in range(L)] for m in range(M)] for j in range(J)]
        task = [[mdl.interval_var(size = p[j][m], name = "J{}-M{}".format(j,m)) for m in range(M)] for j in range(J)]
        
        sequences = [[sequence_var([tasks[j][m][l] for j in range(J)], name='M{}-L{}'.format(m,l)) for l in range(L)] for m in range(M)]
        
        for j in range(J):
            for m in range(1,M):
                for l in range(L):
                    mdl.add(mdl.presence_of(tasks[j][m-1][l]) == mdl.presence_of(tasks[j][m][l]))

        # # chose which factory executes the job's operation
        for j in range(J):
            for m in range(M):
                mdl.add(mdl.alternative(task[j][m], [tasks[j][m][l] for l in range(L)]))
        
        M_no_3_4 = [0, 1, 4, 5]
        
        # Production line
        # for m in M_no_3_4:
        #     mdl.add(mdl.no_overlap(task[j][m] for j in range(J)))
        for l in range(L):
            for m in M_no_3_4:
                mdl.add(mdl.no_overlap(tasks[j][m][l] for j in range(J)))

        mdl.add(mdl.no_overlap(task[j][2] for j in range(J)))
        
        
        # the next process will start after this process be executed immediately
        for j in range(J):
            for m in range(1,M):
                for l in range(L):
                    mdl.add(mdl.end_before_start(task[j][m-1], task[j][m]))
        
        # Force sequences to be all identical on all machines
        for l in range(L):
            for m in range(1,M):
                mdl.add(same_sequence(sequences[m-1][l], sequences[m][l]))
        
        
        mdl.add(mdl.minimize(mdl.max(mdl.end_of(task[j][M-1]) for j in range(J))))
        
        print('Solving model...')
        res = mdl.solve(TimeLimit=timeLimit)
        print('Solution: ')
        res.print_solution()
        
        # Remaining parameters
        opt_value = res.get_objective_values()
        obj_bound = res.get_objective_bounds()
        obj_gap = res.get_objective_gaps()
        total_solve_time = res.get_solve_time()
        # interval_sol = msol.get_value()
        # print(" the interval_sol is ", interval_sol)
        
        print("type of opt_value in Python is", opt_value)
        
        return opt_value, total_solve_time, obj_bound

    """


    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    best_sub_obj_value = 0
    solve_time = 0
    J = load(filenm, "JJ")
    M = load(filenm, "MM")
    p_all = load(filenm, "pp")
    L = load(filenm, "LL")
    W = load(filenm, "TT")
    lineJob = 0
    tableJob = 0
    count_Line = 0
    count_Table = 0
    println("X = $X")
    for i in 1:J
        if X[i, 1] >= 0.9
            count_Line = count_Line + 1
        end

        if X[i, 2] >= 0.9
            count_Table = count_Table + 1
        end
    end
    p_line = zeros(Int64, count_Line, M)
    p_table = zeros(Int64, count_Table, M)



    for i in 1:J
        if X[i, 1] >= 0.9
            lineJob = lineJob + 1
            p_line[lineJob, :] = p_all[i, :]
        end

        if X[i, 2] >= 0.9
            tableJob = tableJob + 1
            p_table[tableJob, :] = p_all[i, :]
        end
    end

    line_opt_vals, line_solve_time, line_obj_bounds = py"line_CP"(lineJob, M, L, p_line, timeLimit)
    line_opt_val = line_opt_vals[1]
    line_obj_bound = line_obj_bounds[1]

    if tableJob < 11 && W == 8
        table_opt_val, table_solve_time, table_obj_bound = MILP_table(tableJob, M, W, p_table, timeLimit)
    else
        table_opt_vals, table_solve_time, table_obj_bounds = py"table_CP"(tableJob, M, W, p_table, timeLimit)
        table_opt_val = table_opt_vals[1]
        table_obj_bound = table_obj_bounds[1]
    end



    # table_opt_val, table_lower_bound, table_gap, table_solve_time, tableMP_time, tableSP_time = logic_based_Benders_locMP(tableJob, M, W, p_table, timeLimit)

    if line_opt_val > table_opt_val
        best_sub_obj_value = line_opt_val
    else
        best_sub_obj_value = table_opt_val
    end

    if line_obj_bound > table_obj_bound
        best_sub_obj_bound = line_obj_bound
    else
        best_sub_obj_bound = table_obj_bound
    end

    solve_time = line_solve_time + table_solve_time

    return best_sub_obj_value, solve_time, line_solve_time, table_solve_time, best_sub_obj_bound
end


function LBBD(filenm, TIME_LIMIT)
    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    println("J = $J")

    M = load(filenm, "MM")
    println("M = $M")

    p = load(filenm, "pp")
    println("p = $p")

    L = load(filenm, "LL")
    println("L = $L")

    W = load(filenm, "TT")
    println("W = $W")

    E = load(filenm, "SS")
    println("E = $E")

    F = 2
    # Create model
    model = Model(CPLEX.Optimizer)

    # Parameter definition
    Ω = sum(p) # A large number

    # Variables
    @variable(model, U[1:J, 1:F], Bin)  # Which production line or mold table the job is processed on

    @variable(model, B[1:J, 1:M] >= 0) # Start time of job n at operation s
    @variable(model, C[1:J, 1:M] >= 0) # Completion time of job n at operation s
    @variable(model, Cmax >= 0)    # Completion time

    # Objective function
    @objective(model, Min, Cmax)

    # Constraints
    @constraint(model, [i in 1:J], sum(U[i, f] for f in 1:F) == 1)
    @constraint(model, [f in 1:F], sum(U[i, f] for i in 1:J) >= 1)
    @constraint(model, [i in 1:J], U[i, 2] - E[i] >= 0)



    ## Basic relationship between completion time and processing start time 
    @constraint(model, [i in 1:J, m in 2:M], C[i, m] >= C[i, m-1] + p[i, m])
    @constraint(model, [i in 1:J], C[i, 1] >= p[i, 1])


    @constraint(model, [i in 1:J], Cmax >= C[i, 6])


    cuts = []
    precision = 1e-6
    best_value = Inf
    timeMasterProblem = 0
    timeSubProblem = 0
    timeSubLine = 0
    timeSubTable = 0
    iterate = 0
    while true
        iterate = iterate + 1
        # Step 1: Solve the master problem
        JuMP.set_time_limit_sec(model, 60)
        JuMP.optimize!(model)
        timeMasterProblem = JuMP.solve_time(model) + timeMasterProblem
        U1 = JuMP.value.(U)
        lower_bound = JuMP.objective_bound(model)

        # Step 2: Solve the subproblem
        upper_bound, subSolTime, subLineSolTime, subTableSolTime, subObjBound = subproblem_solve_Unit(filenm, 30, U1)
        timeSubProblem = subSolTime + timeSubProblem
        timeSubLine = subLineSolTime + timeSubLine
        timeSubTable = subTableSolTime + timeSubTable
        # Step 3: Check the optimality and break the loop
        if upper_bound < best_value
            best_value = upper_bound
        end

        if best_value - lower_bound <= precision
            println("Optimal solution found")
            println("Objective value: ", best_value)
            println("Iteration: ", iterate)
            return best_value, lower_bound, (best_value - lower_bound) / best_value, iterate, timeMasterProblem, timeSubProblem, timeSubLine, timeSubTable
            break
        end

        # Step 4: Add the Benders cut
        cut = @constraint(model, Cmax >= subObjBound * (1 - sum(U[i, f] * (1 - U1[i, f]) for i in 1:J for f in 1:F)))
        push!(cuts, cut)

        if timeMasterProblem + timeSubProblem >= TIME_LIMIT
            println("Time limit reached")
            println("Iteration: ", iterate)
            println("lower bound: ", lower_bound)
            println("upper bound: ", best_value)
            gap = (best_value - lower_bound) / best_value
            println("Gap: ", gap)
            return best_value, lower_bound, (best_value - lower_bound) / best_value, iterate, timeMasterProblem, timeSubProblem, timeSubLine, timeSubTable
            break
        end
    end
end


function LBBD_GBWR(filenm, TIME_LIMIT)
    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    println("J = $J")

    M = load(filenm, "MM")
    println("M = $M")

    p = load(filenm, "pp")
    println("p = $p")

    L = load(filenm, "LL")
    println("L = $L")

    W = load(filenm, "TT")
    println("W = $W")

    E = load(filenm, "SS")
    println("E = $E")

    F = 2
    # Create model
    model = Model(CPLEX.Optimizer)

    # Parameter definition
    Ω = sum(p) # A large number

    # Variables
    @variable(model, U[1:J, 1:F], Bin)  # Which production line or mold table the job is processed on

    @variable(model, B[1:J, 1:M] >= 0) # Start time of job n at operation s
    @variable(model, C[1:J, 1:M] >= 0) # Completion time of job n at operation s
    @variable(model, Cmax >= 0)    # Completion time

    # Objective function
    @objective(model, Min, Cmax)

    # Constraints
    @constraint(model, [i in 1:J], sum(U[i, f] for f in 1:F) == 1)
    @constraint(model, [f in 1:F], sum(U[i, f] for i in 1:J) >= 1)
    @constraint(model, [i in 1:J], U[i, 2] - E[i] >= 0)



    ## Basic relationship between completion time and processing start time 
    @constraint(model, [i in 1:J, m in 2:M], C[i, m] >= C[i, m-1] + p[i, m])
    @constraint(model, [i in 1:J], C[i, 1] >= p[i, 1])


    M_1_2 = [1, 2]
    M_4_5_6 = [4, 5, 6]
    O1 = minimum(sum(p[i, m] for m in M_1_2) for i in 1:J)
    O2 = minimum(sum(p[i, m] for m in M_4_5_6) for i in 1:J)

    @constraint(model, Cmax >= sum(p[i, 3] for i in 1:J) / F + O1 + O2)



    @constraint(model, [i in 1:J], Cmax >= C[i, 6])


    cuts = []
    precision = 1e-6
    best_value = Inf
    timeMasterProblem = 0
    timeSubProblem = 0
    timeSubLine = 0
    timeSubTable = 0
    iterate = 0
    while true
        iterate = iterate + 1
        # Step 1: Solve the master problem
        JuMP.set_time_limit_sec(model, 60)
        JuMP.optimize!(model)
        timeMasterProblem = JuMP.solve_time(model) + timeMasterProblem
        U1 = JuMP.value.(U)
        lower_bound = JuMP.objective_bound(model)

        # Step 2: Solve the subproblem
        upper_bound, subSolTime, subLineSolTime, subTableSolTime, subObjBound = subproblem_solve_Unit(filenm, 30, U1)
        timeSubProblem = subSolTime + timeSubProblem
        timeSubLine = subLineSolTime + timeSubLine
        timeSubTable = subTableSolTime + timeSubTable
        # Step 3: Check the optimality and break the loop
        if upper_bound < best_value
            best_value = upper_bound
        end

        if best_value - lower_bound <= precision
            println("Optimal solution found")
            println("Objective value: ", best_value)
            println("Iteration: ", iterate)
            return best_value, lower_bound, (best_value - lower_bound) / best_value, iterate, timeMasterProblem, timeSubProblem, timeSubLine, timeSubTable
            break
        end

        # Step 4: Add the Benders cut
        cut = @constraint(model, Cmax >= subObjBound * (1 - sum(U[i, f] * (1 - U1[i, f]) for i in 1:J for f in 1:F)))
        push!(cuts, cut)

        if timeMasterProblem + timeSubProblem >= TIME_LIMIT
            println("Time limit reached")
            println("Iteration: ", iterate)
            println("lower bound: ", lower_bound)
            println("upper bound: ", best_value)
            gap = (best_value - lower_bound) / best_value
            println("Gap: ", gap)
            return best_value, lower_bound, (best_value - lower_bound) / best_value, iterate, timeMasterProblem, timeSubProblem, timeSubLine, timeSubTable
            break
        end
    end
end

function LBBD_ADBWR(filenm, TIME_LIMIT)
    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    println("J = $J")

    M = load(filenm, "MM")
    println("M = $M")

    p = load(filenm, "pp")
    println("p = $p")

    L = load(filenm, "LL")
    println("L = $L")

    W = load(filenm, "TT")
    println("W = $W")

    S = load(filenm, "SS")
    println("S = $S")

    F = 2

    model = Model(CPLEX.Optimizer)


    Ω = sum(p)

    @variable(model, U[1:J, 1:F], Bin)

    @variable(model, B[1:J, 1:M] >= 0)
    @variable(model, C[1:J, 1:M] >= 0)
    @variable(model, Cmax >= 0)


    @objective(model, Min, Cmax)


    @constraint(model, [i in 1:J], sum(U[i, f] for f in 1:F) == 1)
    @constraint(model, [f in 1:F], sum(U[i, f] for i in 1:J) >= 1)
    @constraint(model, [i in 1:J], U[i, 2] - S[i] >= 0)


    @constraint(model, [i in 1:J, m in 2:M], C[i, m] >= C[i, m-1] + p[i, m])
    @constraint(model, [i in 1:J], C[i, 1] >= p[i, 1])

    M_1_2 = [1, 2]
    M_4_5_6 = [4, 5, 6]

    @variable(model, R1[1:F] >= 0)
    @variable(model, R2[1:F] >= 0)
    @variable(model, v[1:J, 1:F], Bin)
    @variable(model, q[1:J, 1:F], Bin)
    @constraint(model, [f in 1:F], sum(v[i, f] for i in 1:J) >= 1)
    @constraint(model, [i in 1:J, f in 1:F], v[i, f] <= U[i, f])
    @constraint(model, [f in 1:F], sum(q[i, f] for i in 1:J) >= 1)
    @constraint(model, [i in 1:J, f in 1:F], q[i, f] <= U[i, f])

    @constraint(model, [i in 1:J, f in 1:F], R1[f] <= sum(p[i, m] for m in M_1_2) + Ω * (1 - U[i, f]))
    @constraint(model, [i in 1:J, f in 1:F], R1[f] >= sum(p[i, m] for m in M_1_2) - Ω * (1 - v[i, f]) - Ω * (1 - U[i, f]))
    @constraint(model, [i in 1:J, f in 1:F], R2[f] <= sum(p[i, m] for m in M_4_5_6) + Ω * (1 - U[i, f]))
    @constraint(model, [i in 1:J, f in 1:F], R2[f] >= sum(p[i, m] for m in M_4_5_6) - Ω * (1 - q[i, f]) - Ω * (1 - U[i, f]))

    @variable(model, N >= 0)
    @variable(model, s, Bin)
    @constraint(model, [f in 1:F], N <= R1[f] + R2[f])
    @constraint(model, N >= R1[1] + R2[1] - H * s)
    @constraint(model, N >= R1[2] + R2[2] - H * (1 - s))

    @constraint(model, Cmax >= sum(p[i, 3] for i in 1:J) / F + N)

    @constraint(model, [i in 1:J], Cmax >= C[i, 6])

    cuts = []
    precision = 1e-6
    best_value = Inf
    timeMasterProblem = 0
    timeSubProblem = 0
    timeSubLine = 0
    timeSubTable = 0
    iterate = 0
    while true
        iterate = iterate + 1
        # Step 1: Solve the master problem
        JuMP.set_time_limit_sec(model, 60)
        JuMP.optimize!(model)
        timeMasterProblem = JuMP.solve_time(model) + timeMasterProblem
        U1 = JuMP.value.(U)
        lower_bound = JuMP.objective_bound(model)

        # Step 2: Solve the subproblem
        upper_bound, subSolTime, subLineSolTime, subTableSolTime, subObjBound = subproblem_solve_Unit(filenm, 30, U1)
        timeSubProblem = subSolTime + timeSubProblem
        timeSubLine = subLineSolTime + timeSubLine
        timeSubTable = subTableSolTime + timeSubTable
        # Step 3: Check the optimality and break the loop
        if upper_bound < best_value
            best_value = upper_bound
        end

        if best_value - lower_bound <= precision
            println("Optimal solution found")
            println("Objective value: ", best_value)
            println("Iteration: ", iterate)
            return best_value, lower_bound, (best_value - lower_bound) / best_value, iterate, timeMasterProblem, timeSubProblem, timeSubLine, timeSubTable
            break
        end

        # Step 4: Add the Benders cut
        cut = @constraint(model, Cmax >= subObjBound * (1 - sum(U[i, f] * (1 - U1[i, f]) for i in 1:J for f in 1:F)))
        push!(cuts, cut)

        if timeMasterProblem + timeSubProblem >= TIME_LIMIT
            println("Time limit reached")
            println("Iteration: ", iterate)
            println("lower bound: ", lower_bound)
            println("upper bound: ", best_value)
            gap = (best_value - lower_bound) / best_value
            println("Gap: ", gap)
            return best_value, lower_bound, (best_value - lower_bound) / best_value, iterate, timeMasterProblem, timeSubProblem, timeSubLine, timeSubTable
            break
        end
    end
end


function LBBD_LBWR(filenm, TIME_LIMIT)
    #-----------------------------------------------------------------------------
    # Initialize Data
    #-----------------------------------------------------------------------------
    J = load(filenm, "JJ")
    println("J = $J")

    M = load(filenm, "MM")
    println("M = $M")

    p = load(filenm, "pp")
    println("p = $p")

    L = load(filenm, "LL")
    println("L = $L")

    W = load(filenm, "TT")
    println("W = $W")

    E = load(filenm, "SS")
    println("E = $E")

    F = 2
    # Create model
    model = Model(CPLEX.Optimizer)

    # Parameter definition
    Ω = sum(p) # A large number

    # Variables
    @variable(model, U[1:J, 1:F], Bin)  # Which production line or mold table the job is processed on

    @variable(model, B[1:J, 1:M] >= 0) # Start time of job n at operation s
    @variable(model, C[1:J, 1:M] >= 0) # Completion time of job n at operation s
    @variable(model, Cmax >= 0)    # Completion time


    @objective(model, Min, Cmax)


    @constraint(model, [i in 1:J], sum(U[i, f] for f in 1:F) == 1)
    @constraint(model, [i in 1:J], U[i, 2] - E[i] >= 0)

    @constraint(model, [i in 1:J, m in 2:M], C[i, m] >= C[i, m-1] + p[i, m])
    @constraint(model, [i in 1:J], C[i, 1] >= p[i, 1])

    M_1_2 = [1, 2]
    M_4_5_6 = [4, 5, 6]

    @variable(model, R1[1:F] >= 0)
    @variable(model, R2[1:F] >= 0)
    @variable(model, v[1:J, 1:F], Bin)
    @variable(model, q[1:J, 1:F], Bin)
    @constraint(model, [f in 1:F], sum(v[i, f] for i in 1:J) >= 1)
    @constraint(model, [i in 1:J, f in 1:F], v[i, f] <= U[i, f])
    @constraint(model, [f in 1:F], sum(q[i, f] for i in 1:J) >= 1)
    @constraint(model, [i in 1:J, f in 1:F], q[i, f] <= U[i, f])
    @constraint(model, [i in 1:J, f in 1:F], R1[f] <= sum(p[i, m] for m in M_1_2) + Ω * (1 - U[i, f]))
    @constraint(model, [i in 1:J, f in 1:F], R1[f] >= sum(p[i, m] for m in M_1_2) - Ω * (1 - v[i, f]) - Ω * (1 - U[i, f]))
    @constraint(model, [i in 1:J, f in 1:F], R2[f] <= sum(p[i, m] for m in M_4_5_6) + Ω * (1 - U[i, f]))
    @constraint(model, [i in 1:J, f in 1:F], R2[f] >= sum(p[i, m] for m in M_4_5_6) - Ω * (1 - q[i, f]) - Ω * (1 - U[i, f]))
    @constraint(model, [f in 1:F], Cmax >= sum(p[i, 3] * U[i, f] for i in 1:J) + R1[f] + R2[f])

    @constraint(model, [i in 1:J], Cmax >= C[i, 6])


    cuts = []
    precision = 1e-6
    best_value = Inf
    timeMasterProblem = 0
    timeSubProblem = 0
    timeSubLine = 0
    timeSubTable = 0
    iterate = 0
    while true
        iterate = iterate + 1
        # Step 1: Solve the master problem
        JuMP.set_time_limit_sec(model, 60)
        JuMP.optimize!(model)
        timeMasterProblem = JuMP.solve_time(model) + timeMasterProblem
        U1 = JuMP.value.(U)
        lower_bound = JuMP.objective_bound(model)

        # Step 2: Solve the subproblem
        upper_bound, subSolTime, subLineSolTime, subTableSolTime, subObjBound = subproblem_solve_Unit(filenm, 30, U1)
        timeSubProblem = subSolTime + timeSubProblem
        timeSubLine = subLineSolTime + timeSubLine
        timeSubTable = subTableSolTime + timeSubTable
        # Step 3: Check the optimality and break the loop
        if upper_bound < best_value
            best_value = upper_bound
        end

        if best_value - lower_bound <= precision
            println("Optimal solution found")
            println("Objective value: ", best_value)
            println("Iteration: ", iterate)
            return best_value, lower_bound, (best_value - lower_bound) / best_value, iterate, timeMasterProblem, timeSubProblem, timeSubLine, timeSubTable
            break
        end

        # Step 4: Add the Benders cut
        cut = @constraint(model, Cmax >= subObjBound * (1 - sum(U[i, f] * (1 - U1[i, f]) for i in 1:J for f in 1:F)))
        push!(cuts, cut)

        if timeMasterProblem + timeSubProblem >= TIME_LIMIT
            println("Time limit reached")
            println("Iteration: ", iterate)
            println("lower bound: ", lower_bound)
            println("upper bound: ", best_value)
            gap = (best_value - lower_bound) / best_value
            println("Gap: ", gap)
            return best_value, lower_bound, (best_value - lower_bound) / best_value, iterate, timeMasterProblem, timeSubProblem, timeSubLine, timeSubTable
            break
        end
    end
end

# ---------------------------------- Data Generation --------------------------------------

function Data_Generation(JOB, MACHINE, LINE_NUM, TABLE_NUM, PROCESSING_TIME_4, index)

    p = rand(10:30, JOB, MACHINE)    # processing time

    SPECIAL_JOB_NUM = 0
    SPECIAL_JOB = Array{Bool}(undef, JOB)
    # Initialize SPECIAL_JOB
    for j in 1:JOB
        SPECIAL_JOB[j] = 0
    end
    for j in 1:JOB
        if rand() < 0.2
            SPECIAL_JOB[j] = 1
            SPECIAL_JOB_NUM = SPECIAL_JOB_NUM + 1
        end
    end

    println("SPECIAL_JOB = $SPECIAL_JOB")
    for j in 1:JOB
        p[j, 4] = PROCESSING_TIME_4
    end
    p6 = rand(5:10, JOB)
    for j in 1:JOB
        p[j, 6] = p6[j]
    end

    println("p = $p")
    filenm = join(["Ins_", string(JOB), "_", string(MACHINE), "_", string(LINE_NUM), "_", string(TABLE_NUM), "_", string(index), "_", string(SPECIAL_JOB_NUM), ".jld"])
    save(filenm, "JJ", JOB, "MM", MACHINE, "LL", LINE_NUM, "TT", TABLE_NUM, "pp", p, "SS", SPECIAL_JOB)
end