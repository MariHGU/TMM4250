import math
import numpy as np


#Problem 2
def solveArchLength(model, archLength=0.02, max_steps=50, max_iter=30):
    num_dofs = model.get_num_dofs()
    uVec = np.zeros(num_dofs)
    Lambda = 0.0

    # Free degrees of freedom (excluding boundary conditions)
    free_dofs = np.setdiff1d(
        np.arange(num_dofs), model.bc - 1
    )

    q_Vec = model.get_incremental_load(Lambda)[free_dofs]

    # Previous predictor direction, including load factor
    d_q_prev = None

    # Store initial equilibrium state
    model.append_solution(Lambda, uVec)

    for iStep in range(max_steps):

        # Save previous converged state
        u_start = uVec.copy()
        Lambda_start = Lambda

        # Predictor step (Eq. 3.2.6 - 3.2.8)
        K_mat = model.get_K_sys(uVec)
        K_free = K_mat[np.ix_(free_dofs, free_dofs)]

        w_q0 = np.linalg.solve(K_free, q_Vec)

        # Augmented predictor direction
        direction = np.append(w_q0, 1.0)
        f = np.linalg.norm(direction)

        d_Lambda = archLength / f

        # Choose sign from previous predictor direction
        if d_q_prev is not None:
            if np.dot(d_q_prev, direction) < 0:
                d_Lambda = -d_Lambda

        d_u = d_Lambda * w_q0

        uVec[free_dofs] += d_u
        Lambda += d_Lambda

        # Save predictor direction for next step
        d_q_prev = d_Lambda * direction

        # Corrector iterations (normal plane method)
        converged = False

        for iIter in range(max_iter):

            res_Vec = model.get_residual(Lambda, uVec)
            res_free = res_Vec[free_dofs]

            # Check force equilibrium
            tolerance = 1.0e-8 * max(
                1.0, np.linalg.norm(q_Vec)
            )

            if np.linalg.norm(res_free) < tolerance:
                converged = True
                break

            # Updated tangent stiffness
            K_mat = model.get_K_sys(uVec)
            K_free = K_mat[np.ix_(free_dofs, free_dofs)]

            # Residual and load direction solutions
            w_r = np.linalg.solve(K_free, res_free)
            w_q = np.linalg.solve(K_free, q_Vec)

            # Normal plane correction (Eq. 3.2.11)
            delta_Lambda = -(
                np.dot(w_q0, w_r)
                / (1.0 + np.dot(w_q0, w_q))
            )

            delta_u = w_r + delta_Lambda * w_q

            # Update displacement and load factor
            uVec[free_dofs] += delta_u
            Lambda += delta_Lambda

        if not converged:
            uVec = u_start
            Lambda = Lambda_start
            raise RuntimeError(
                f"Arc-length step {iStep + 1} did not converge"
            )

        model.append_solution(Lambda, uVec)

        print(
            f"Arc step {iStep + 1}: "
            f"Lambda = {Lambda:.6f}, "
            f"iterations = {iIter + 1}"
        )

#Problem 1
def solveNonlinLoadControl(model, load_steps=0.01, max_steps=100, max_iter=30): 
    num_dofs = model.get_num_dofs()
    uVec   = np.zeros(num_dofs)
    d_uVec = np.zeros(num_dofs)

    for iStep in range(max_steps):

        Lambda = load_steps * iStep

        #Added: Residual at the start of the step
        res_Vec = model.get_residual(Lambda, uVec)

        for iIter in range(max_iter):

            #Added: Newtonstep
            K_mat = model.get_K_sys(uVec)
            d_uVec = np.linalg.solve(K_mat, res_Vec)
            uVec = uVec + d_uVec

            res_Vec = model.get_residual(Lambda, uVec)
            if (res_Vec.dot(res_Vec) < 1.0e-15):
                break



        model.append_solution(Lambda, uVec)
        print("Non-Linear load step {:}  load_factor= {:12.3e}".format(iStep, Lambda))


#Used for linear solution
def solveLinearSteps(model, load_steps=0.01, max_steps=100):
    num_dofs = model.get_num_dofs()
    uVec = np.zeros(num_dofs)

    for iStep in range(max_steps):

        Lambda = load_steps * iStep

        q_Vec   = model.get_incremental_load(Lambda)

        K_mat = model.get_K_sys(uVec)

        d_q = np.linalg.solve(K_mat, q_Vec)

        uVec = d_q * Lambda

        model.append_solution(Lambda, uVec)
        print("Linear load step {:}  load_factor= {:12.3e}".format(iStep, Lambda))

