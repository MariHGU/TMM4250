import BeamModels_with_TODO as models
import SolverAlgorithms_with_TODO as algs

num_nodes = 9
#model = models.SimplySupportedBeamModel(num_nodes)
model = models.CantileverWithEndMoment(num_nodes)

algs.solveLinearSteps(model,load_steps=0.01,max_steps=100)
#algs.solveNonlinLoadControl(model,load_steps=0.01,max_steps=100)
#algs.solveArchLength(model,load_steps=0.01,max_steps=100)

num_steps = len(model.load_history)

for iStep in range(num_steps):
   print("LoadFactor= {:12.3e}".format(model.load_history[iStep]))
   print("dispVec={:}".format(iStep))
   print(model.disp_history[iStep])

# Create matplotlib plots
step_inc = (num_steps // 10)
for iStep in range(0,len(model.load_history), step_inc):
    model.plotDispState(iStep)

# write vtu-files for animation in ParaView
for iStep in range(num_steps):
    model.vtu_print_state(iStep)
print("End")
