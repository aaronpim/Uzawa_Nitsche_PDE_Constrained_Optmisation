import imageio.v2 as imageio
from pdf2image import convert_from_path
state = []
control = []
adjnt = []
for epoch in range(0,5000,10):
    statepng = convert_from_path(f'./Vary_alpha_0.0001_eps_0.1/state_frame_{epoch}.pdf')
    statepng[0].save(f'./Vary_alpha_0.0001_eps_0.1/state_{epoch}.png', 'PNG')
    state.append(imageio.imread(f'./Vary_alpha_0.0001_eps_0.1/state_{epoch}.png'))

    contrpng = convert_from_path(f'./Vary_alpha_0.0001_eps_0.1/control_frame_{epoch}.pdf')
    contrpng[0].save(f'./Vary_alpha_0.0001_eps_0.1/control_{epoch}.png', 'PNG')
    control.append(imageio.imread(f'./Vary_alpha_0.0001_eps_0.1/control_{epoch}.png'))

    adjntpng = convert_from_path(f'./Vary_alpha_0.0001_eps_0.1/adjoint_frame_{epoch}.pdf')
    adjntpng[0].save(f'./Vary_alpha_0.0001_eps_0.1/adjoint_{epoch}.png', 'PNG')
    adjnt.append(imageio.imread(f'./Vary_alpha_0.0001_eps_0.1/adjoint_{epoch}.png'))
    
imageio.mimsave('./state_Vary_alpha_0.0001_eps_0.1.gif', state)
imageio.mimsave('./control_Vary_alpha_0.0001_eps_0.1.gif', control)
imageio.mimsave('./adjoint_Vary_alpha_0.0001_eps_0.1.gif', adjnt)
