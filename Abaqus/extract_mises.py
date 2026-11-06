
from odbAccess import openOdb

# Configuration
odb_path = 'sc-00-init-IR-100y.odb'
step_name = 'Step-3'
output_file = 'max_mises.csv'

odb = openOdb(path=odb_path, readOnly=True)
step = odb.steps[step_name]

with open(output_file, 'w') as file:
    file.write('Frame,Time,Max_Mises,Instance,Element\n')

    for i, frame in enumerate(step.frames):

        if 'S' not in frame.fieldOutputs:
            continue

        stress = frame.fieldOutputs['S']

        if not stress.values:
            continue

        max_value = max(stress.values, key=lambda v: v.mises)

        file.write('{},{},{},{},{}\n'.format(
            i,
            frame.frameValue,
            max_value.mises,
            max_value.instance.name,
            max_value.elementLabel
        ))

odb.close()

print('Results exported to', output_file)
